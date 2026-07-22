#!/usr/bin/env python3
import cv2
import numpy as np
import logging
import json
import csv
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union

try:
    from scipy.signal import savgol_filter
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

log = logging.getLogger(__name__)


class ROIRetimeMethod:

    def __init__(
        self,
        roi_config: Union[str, Path],
        dt_range: int = 8,
        dx_range: int = 120,
        dy_range: int = 40,
        work_scale: float = 0.25,
        transform_mode: str = 'retime+translate',
        savgol_window: int = 11,
        savgol_poly: int = 3,
        min_confidence: float = 0.15,
        debug_frames: Optional[List[int]] = None,
        static_pre_correction: Optional[dict] = None,
    ):
        with open(roi_config, 'r') as f:
            cfg = json.load(f)
        self.rois = cfg['rois']
        self.dt_range = dt_range
        self.dx_range = dx_range
        self.dy_range = dy_range
        self.work_scale = work_scale
        self.transform_mode = transform_mode
        self.savgol_window = savgol_window
        self.savgol_poly = savgol_poly
        self.min_confidence = min_confidence
        self.debug_frames = debug_frames
        self.static_pre_correction = static_pre_correction

    def _structural(self, img_bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        gray_blur = cv2.GaussianBlur(gray, (0, 0), 2.0)
        sx = cv2.Sobel(gray_blur, cv2.CV_32F, 1, 0, ksize=3)
        sy = cv2.Sobel(gray_blur, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(sx, sy)
        mag = cv2.GaussianBlur(mag, (0, 0), 3.0)
        mag = cv2.normalize(mag, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        return mag

    def _apodize(self, patch: np.ndarray) -> np.ndarray:
        rows, cols = patch.shape[:2]
        win = np.outer(np.hanning(rows), np.hanning(cols)).astype(np.float32)
        return patch * win

    def _apply_static_correction(self, frame: np.ndarray, corr: dict) -> np.ndarray:
        H, W = frame.shape[:2]
        s = float(corr.get('scale', 1.0))
        cx = float(corr.get('cx', W / 2.0))
        cy = float(corr.get('cy', H / 2.0))
        tx = float(corr.get('tx', 0.0))
        ty = float(corr.get('ty', 0.0))
        # Scale around (cx, cy) then translate: maps src→dst so warpAffine needs no inversion flag
        M = np.float32([
            [s, 0.0, cx * (1.0 - s) + tx],
            [0.0, s, cy * (1.0 - s) + ty],
        ])
        return cv2.warpAffine(frame, M, (W, H),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)

    def _search_roi(self, orig_struct_small: np.ndarray,
                    clean_structs_small: List[np.ndarray],
                    roi: dict, t: int, N: int) -> dict:
        scale = self.work_scale
        H_small, W_small = orig_struct_small.shape[:2]
        x0 = max(0, min(int(roi['x'] * scale + 0.5), W_small - 2))
        y0 = max(0, min(int(roi['y'] * scale + 0.5), H_small - 2))
        x1 = max(x0 + 1, min(x0 + int(roi['w'] * scale + 0.5), W_small))
        y1 = max(y0 + 1, min(y0 + int(roi['h'] * scale + 0.5), H_small))
        template = orig_struct_small[y0:y1, x0:x1]
        if template.shape[0] < 4 or template.shape[1] < 4:
            return {'dt': 0, 'dx': 0.0, 'dy': 0.0, 'conf': 0.0, 'all_candidates': []}

        pad_x = int(self.dx_range * scale)
        pad_y = int(self.dy_range * scale)
        template_u8 = (self._apodize(template.copy()) * 255).clip(0, 255).astype(np.uint8)
        candidates = []

        for dt in range(-self.dt_range, self.dt_range + 1):
            t_clean = max(0, min(N - 1, t + dt))
            search_frame = clean_structs_small[t_clean]
            sx0 = max(0, x0 - pad_x)
            sy0 = max(0, y0 - pad_y)
            sx1 = min(W_small, x1 + pad_x)
            sy1 = min(H_small, y1 + pad_y)
            if (sx1 - sx0) < template.shape[1] or (sy1 - sy0) < template.shape[0]:
                continue
            search_region = search_frame[sy0:sy1, sx0:sx1]
            pad_x_actual = x0 - sx0
            pad_y_actual = y0 - sy0
            search_u8 = (search_region * 255).clip(0, 255).astype(np.uint8)
            result_map = cv2.matchTemplate(search_u8, template_u8, cv2.TM_CCOEFF_NORMED)
            _, best_score, _, best_loc = cv2.minMaxLoc(result_map)
            dx_small = best_loc[0] - pad_x_actual
            dy_small = best_loc[1] - pad_y_actual
            candidates.append({
                'dt': dt,
                'dx': float(dx_small / scale),
                'dy': float(dy_small / scale),
                'ncc': float(best_score),
                'result_map': result_map,
                'best_loc': best_loc,
            })

        if not candidates:
            return {'dt': 0, 'dx': 0.0, 'dy': 0.0, 'conf': 0.0, 'all_candidates': []}
        best = max(candidates, key=lambda c: c['ncc'])
        return {
            'dt': best['dt'],
            'dx': best['dx'],
            'dy': best['dy'],
            'conf': best['ncc'],
            'all_candidates': candidates,
        }

    def _combine_rois(self, roi_results: List[dict], rois: List[dict]) -> dict:
        def weighted_median(values, weights):
            if not values or sum(weights) < 1e-12:
                return 0.0
            sorted_pairs = sorted(zip(values, weights), key=lambda x: x[0])
            sv = [p[0] for p in sorted_pairs]
            sw = [p[1] for p in sorted_pairs]
            cumsum = np.cumsum(sw)
            idx = np.searchsorted(cumsum, cumsum[-1] / 2.0, side='left')
            return float(sv[min(idx, len(sv) - 1)])

        # zip keeps result<->roi-config aligned so weight lookup is correct
        pairs = list(zip(roi_results, rois))
        conf_pairs = [(res, r) for res, r in pairs if res['conf'] >= self.min_confidence]
        if not conf_pairs:
            conf_pairs = [max(pairs, key=lambda p: p[0]['conf'])]

        f_dt = [p[0]['dt'] for p in conf_pairs]
        f_dx = [p[0]['dx'] for p in conf_pairs]
        f_dy = [p[0]['dy'] for p in conf_pairs]
        f_weights = [p[0]['conf'] * p[1].get('weight', 1.0) for p in conf_pairs]

        if sum(f_weights) < 1e-9:
            return {'dt': 0, 'dx': 0.0, 'dy': 0.0, 'conf': 0.0, 'agreement': 0.0}

        dt_med = int(round(weighted_median(f_dt, f_weights)))
        dx_med = weighted_median(f_dx, f_weights)
        dy_med = weighted_median(f_dy, f_weights)
        mean_conf = float(np.average(
            [p[0]['conf'] for p in conf_pairs],
            weights=[p[1].get('weight', 1.0) for p in conf_pairs]
        ))
        agreement = 1.0 / (1.0 + float(np.std(f_dx))) if len(f_dx) > 1 else 1.0
        return {'dt': dt_med, 'dx': dx_med, 'dy': dy_med,
                'conf': mean_conf, 'agreement': agreement}

    def _smooth_curves(self, raw: List[dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        N = len(raw)
        dt_arr = np.array([r['dt'] for r in raw], dtype=float)
        dx_arr = np.array([r['dx'] for r in raw], dtype=float)
        dy_arr = np.array([r['dy'] for r in raw], dtype=float)
        conf_arr = np.array([r['conf'] for r in raw], dtype=float)

        mask = conf_arr < self.min_confidence
        if mask.any():
            xp = np.where(~mask)[0]
            if len(xp) > 0:
                x = np.arange(N)
                for arr in [dt_arr, dx_arr, dy_arr]:
                    arr[mask] = np.interp(x[mask], xp, arr[xp])

        window = max(3, min(self.savgol_window, N if N % 2 == 1 else N - 1))
        poly = min(self.savgol_poly, window - 1)

        if _HAS_SCIPY:
            dt_s = savgol_filter(dt_arr, window_length=window, polyorder=poly)
            dx_s = savgol_filter(dx_arr, window_length=window, polyorder=poly)
            dy_s = savgol_filter(dy_arr, window_length=window, polyorder=poly)
        else:
            log.warning('scipy not available; using uniform convolution for smoothing')
            kernel = np.ones(window) / window
            dt_s = np.convolve(np.pad(dt_arr, window // 2, mode='edge'), kernel, mode='valid')[:N]
            dx_s = np.convolve(np.pad(dx_arr, window // 2, mode='edge'), kernel, mode='valid')[:N]
            dy_s = np.convolve(np.pad(dy_arr, window // 2, mode='edge'), kernel, mode='valid')[:N]

        return dt_s, dx_s, dy_s

    def _apply_transform(self, clean_frames: List[np.ndarray],
                         dt_smooth: np.ndarray, dx_smooth: np.ndarray,
                         dy_smooth: np.ndarray) -> List[np.ndarray]:
        N = len(clean_frames)
        warped = []
        for t in range(N):
            dt_val = float(dt_smooth[t])
            t0_c = int(np.floor(t + dt_val))
            t1_c = t0_c + 1
            alpha = (t + dt_val) - t0_c
            t0_c = max(0, min(N - 1, t0_c))
            t1_c = max(0, min(N - 1, t1_c))
            resampled = np.clip(
                (1 - alpha) * clean_frames[t0_c].astype(float) +
                alpha * clean_frames[t1_c].astype(float),
                0, 255
            ).astype(np.uint8)
            if self.transform_mode == 'retime_only':
                warped.append(resampled)
            else:
                H, W = resampled.shape[:2]
                M = np.float32([[1, 0, dx_smooth[t]], [0, 1, dy_smooth[t]]])
                warped.append(cv2.warpAffine(resampled, M, (W, H),
                                             flags=cv2.INTER_LINEAR,
                                             borderMode=cv2.BORDER_REPLICATE))
        return warped

    def _write_roi_debug(self, out_dir: Path, t: int, orig_bgr: np.ndarray,
                         clean_bgr: np.ndarray, roi: dict, roi_result: dict):
        out_dir.mkdir(parents=True, exist_ok=True)
        H_img, W_img = orig_bgr.shape[:2]
        x = max(0, min(roi['x'], W_img - 1))
        y = max(0, min(roi['y'], H_img - 1))
        x1 = min(x + roi['w'], W_img)
        y1 = min(y + roi['h'], H_img)
        orig_crop = orig_bgr[y:y1, x:x1]
        clean_crop = clean_bgr[y:y1, x:x1]
        if orig_crop.size == 0:
            return

        orig_edges = cv2.Canny(cv2.cvtColor(orig_crop, cv2.COLOR_BGR2GRAY), 30, 80)
        clean_edges = cv2.Canny(cv2.cvtColor(clean_crop, cv2.COLOR_BGR2GRAY), 30, 80)
        overlay = np.full((*orig_crop.shape[:2], 3), 30, dtype=np.uint8)
        overlay[:, :, 1] = orig_edges   # green = original
        overlay[:, :, 2] = clean_edges  # red = clean

        max_w = 640
        h_ov, w_ov = overlay.shape[:2]
        if w_ov > max_w:
            overlay = cv2.resize(overlay, (max_w, int(h_ov * max_w / w_ov)))

        ncc_panel = np.full((overlay.shape[0], max_w, 3), 80, dtype=np.uint8)
        candidates = roi_result.get('all_candidates', [])
        if candidates:
            best_cand = max(candidates, key=lambda c: c['ncc'])
            rmap = best_cand.get('result_map')
            if rmap is not None and isinstance(rmap, np.ndarray):
                rmap_u8 = cv2.normalize(rmap, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                rmap_color = cv2.applyColorMap(rmap_u8, cv2.COLORMAP_JET)
                h_r, w_r = rmap_color.shape[:2]
                scale_fit = min(overlay.shape[0] / max(h_r, 1), max_w / max(w_r, 1))
                rmap_color = cv2.resize(rmap_color,
                                        (int(w_r * scale_fit), int(h_r * scale_fit)))
                ph, pw = rmap_color.shape[:2]
                ncc_panel[:ph, :pw] = rmap_color
                loc = best_cand.get('best_loc')
                if loc is not None:
                    cx = int(loc[0] * scale_fit)
                    cy = int(loc[1] * scale_fit)
                    cv2.circle(ncc_panel, (cx, cy), 3, (255, 255, 255), -1)

        composite = np.hstack([overlay, ncc_panel])
        label = (f"f{t:03d} {roi['name']}  dt={roi_result['dt']:+d}  "
                 f"dx={roi_result['dx']:+.1f}  dy={roi_result['dy']:+.1f}  "
                 f"conf={roi_result['conf']:.3f}")
        cv2.putText(composite, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.imwrite(str(out_dir / f'frame_{t:03d}_{roi["name"]}_candidates.jpg'), composite)

    def _write_curves(self, out_dir: Path, raw: List[dict],
                      dt_s: np.ndarray, dx_s: np.ndarray, dy_s: np.ndarray,
                      roi_results_per_frame: List[dict]):
        curves_dir = out_dir / 'curves'
        curves_dir.mkdir(parents=True, exist_ok=True)
        N = len(raw)
        for key, smooth_arr in [('dt', dt_s), ('dx', dx_s), ('dy', dy_s)]:
            with open(curves_dir / f'{key}_curve.csv', 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['frame', f'raw_{key}', f'smooth_{key}', 'confidence'])
                for i in range(N):
                    w.writerow([i, raw[i][key], smooth_arr[i], raw[i]['conf']])
        roi_names = [r['name'] for r in self.rois]
        with open(curves_dir / 'roi_agreement.csv', 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['frame'] + [f'{n}_dx' for n in roi_names] + ['agreement'])
            for i in range(N):
                row = [i]
                for name in roi_names:
                    row.append(roi_results_per_frame[i].get(name, {}).get('dx', 0.0))
                row.append(raw[i]['agreement'])
                w.writerow(row)

    def _write_crop_preview(self, out_dir: Path, orig_frames: List[np.ndarray],
                            clean_versions: Dict[str, List[np.ndarray]],
                            rois: List[dict], fps: float = 24.0):
        preview_dir = out_dir / 'previews'
        preview_dir.mkdir(parents=True, exist_ok=True)
        H_img, W_img = orig_frames[0].shape[:2]
        max_strip_w = 400

        roi_meta = []
        for roi in rois:
            x = max(0, min(roi['x'], W_img - 1))
            y = max(0, min(roi['y'], H_img - 1))
            x1 = min(x + roi['w'], W_img)
            y1 = min(y + roi['h'], H_img)
            w_roi, h_roi = x1 - x, y1 - y
            if w_roi < 1 or h_roi < 1:
                continue
            roi_meta.append({'x': x, 'y': y, 'x1': x1, 'y1': y1,
                              'name': roi['name'],
                              'strip_h': int(h_roi * max_strip_w / w_roi),
                              'strip_w': max_strip_w})

        if not roi_meta:
            log.warning('No valid ROIs for crop preview')
            return

        total_h = sum(m['strip_h'] for m in roi_meta)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        for version_name, clean_frames in clean_versions.items():
            out_path = preview_dir / f'{version_name}_edge_overlay_crops.mp4'
            writer = cv2.VideoWriter(str(out_path), fourcc, fps, (max_strip_w, total_h))
            if not writer.isOpened():
                log.error('Cannot open VideoWriter: %s', out_path)
                continue
            for orig, clean in zip(orig_frames, clean_frames):
                canvas = np.zeros((total_h, max_strip_w, 3), dtype=np.uint8)
                y_off = 0
                for rm in roi_meta:
                    oc = orig[rm['y']:rm['y1'], rm['x']:rm['x1']]
                    cc = clean[rm['y']:rm['y1'], rm['x']:rm['x1']]
                    oe = cv2.Canny(cv2.cvtColor(oc, cv2.COLOR_BGR2GRAY), 30, 80)
                    ce = cv2.Canny(cv2.cvtColor(cc, cv2.COLOR_BGR2GRAY), 30, 80)
                    strip = np.full((*oc.shape[:2], 3), 30, dtype=np.uint8)
                    strip[:, :, 1] = oe  # green = original
                    strip[:, :, 2] = ce  # red = clean
                    strip = cv2.resize(strip, (rm['strip_w'], rm['strip_h']))
                    canvas[y_off:y_off + rm['strip_h']] = strip
                    y_off += rm['strip_h']
                writer.write(canvas)
            writer.release()
            log.info('Wrote %s', out_path)

    def process_sequence(self, originals: List[np.ndarray],
                         cleans: List[np.ndarray],
                         out_dir=None) -> dict:
        if len(originals) != len(cleans):
            raise ValueError('originals and cleans must have same length')
        N = len(originals)
        if N < 2:
            raise ValueError('need at least 2 frames')

        log.info('Precomputing structural representations at work_scale=%.2f', self.work_scale)
        H, W = originals[0].shape[:2]
        new_w = max(1, int(W * self.work_scale))
        new_h = max(1, int(H * self.work_scale))

        if self.static_pre_correction:
            log.info('Applying static pre-correction: %s', self.static_pre_correction)
            corrected_cleans = [self._apply_static_correction(f, self.static_pre_correction)
                                for f in cleans]
        else:
            corrected_cleans = cleans

        orig_structs = [self._structural(
            cv2.resize(f, (new_w, new_h), interpolation=cv2.INTER_AREA)) for f in originals]
        clean_structs = [self._structural(
            cv2.resize(f, (new_w, new_h), interpolation=cv2.INTER_AREA)) for f in corrected_cleans]

        log.info('Searching per ROI per frame (dt=%d dx=%d dy=%d)...',
                 self.dt_range, self.dx_range, self.dy_range)
        roi_results_per_frame = []
        per_frame_combined = []

        for t in range(N):
            if t % 10 == 0:
                log.info('  frame %d/%d', t, N)
            frame_roi_results = {}
            for roi in self.rois:
                frame_roi_results[roi['name']] = self._search_roi(
                    orig_structs[t], clean_structs, roi, t, N)
            roi_results_per_frame.append(frame_roi_results)

            combined = self._combine_rois(
                [frame_roi_results[r['name']] for r in self.rois], self.rois)
            per_frame_combined.append(combined)
            log.info('f%03d  dt=%+d  dx=%+.1f  dy=%+.1f  conf=%.3f  agree=%.3f',
                     t, combined['dt'], combined['dx'], combined['dy'],
                     combined['conf'], combined['agreement'])

        log.info('Smoothing curves...')
        dt_s, dx_s, dy_s = self._smooth_curves(per_frame_combined)

        log.info('Applying transforms (mode=%s)...', self.transform_mode)
        warped_frames = self._apply_transform(corrected_cleans, dt_s, dx_s, dy_s)
        retime_only = self._apply_transform(corrected_cleans, dt_s, np.zeros(N), np.zeros(N))

        if out_dir is not None:
            out_path = Path(out_dir)
            self._write_curves(out_path, per_frame_combined, dt_s, dx_s, dy_s,
                               roi_results_per_frame)
            dbg = self.debug_frames or [f for f in [0, 15, 30, 63, 100, 125] if f < N]
            for t in dbg:
                for roi in self.rois:
                    self._write_roi_debug(
                        out_path / 'roi_debug', t,
                        originals[t], corrected_cleans[t], roi,
                        roi_results_per_frame[t][roi['name']])
            preview_versions = {'direct': cleans, 'retime': retime_only, 'final': warped_frames}
            if self.static_pre_correction:
                preview_versions['static_corrected'] = corrected_cleans
            self._write_crop_preview(
                out_path, originals, preview_versions, self.rois, fps=24.0)

        flows = []
        for t in range(N):
            flow = np.zeros((H, W, 2), dtype=np.float32)
            flow[:, :, 0] = dx_s[t]
            flow[:, :, 1] = dy_s[t]
            flows.append(flow)

        return {
            'warped_frames': warped_frames,
            'flows': flows,
            'dt_smooth': dt_s.tolist(),
            'dx_smooth': dx_s.tolist(),
            'dy_smooth': dy_s.tolist(),
            'per_frame_combined': per_frame_combined,
            'debug_per_frame': [
                {'model_used': 'translation' if c['conf'] >= self.min_confidence else 'identity',
                 'conf': c['conf'], 'agreement': c['agreement']}
                for c in per_frame_combined
            ],
        }
