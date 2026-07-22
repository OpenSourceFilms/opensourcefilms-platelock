import cv2
import glob as _glob
import numpy as np
from pathlib import Path
import imageio
from tqdm import tqdm
import tempfile
import shutil
import re
from typing import List, Optional, Tuple, Dict


def _natural_sort_key(path: Path) -> list:
    name = path.name
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r"(\d+)", name)]


def load_frames(
    path: str | Path,
    frame_range: Optional[Tuple[int, int]] = None,
    resize: Optional[Tuple[int, int]] = None,
    pixel_format: str = "rgb",
) -> List[np.ndarray]:
    """Return a list of RGB uint8 frames from a video, image directory, or glob pattern."""
    p = Path(path)
    frames: List[np.ndarray] = []

    vid_exts = (".mp4", ".mov", ".avi", ".mkv", ".webm")
    if p.is_dir():
        img_exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".exr")
        img_paths = [f for f in sorted(p.glob("*"), key=_natural_sort_key) if f.suffix.lower() in img_exts]
        if not img_paths:
            # Fall back: directory containing a single video file
            vid_paths = [f for f in sorted(p.glob("*")) if f.suffix.lower() in vid_exts]
            if vid_paths:
                return load_frames(vid_paths[0], frame_range, resize, pixel_format)
            raise FileNotFoundError(f"No images or video found in directory: {path}")
        if frame_range is not None:
            start, end = frame_range
            img_paths = img_paths[start : end + 1]
        frames = [cv2.imread(str(f)) for f in tqdm(img_paths, desc="Loading images", leave=False)]
        frames = [f for f in frames if f is not None]
        frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]

    elif p.is_file() and p.suffix.lower() in vid_exts:
        cap = cv2.VideoCapture(str(p))
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {path}")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        start, end = (0, total - 1) if frame_range is None else (frame_range[0], min(frame_range[1], total - 1))
        current = 0
        pbar = tqdm(total=max(0, end - start + 1), desc="Reading video", leave=False)
        while current <= end:
            ret, frame = cap.read()
            if not ret:
                break
            if current >= start:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                pbar.update(1)
            current += 1
        pbar.close()
        cap.release()

    else:
        matched = sorted([Path(p) for p in _glob.glob(str(path))], key=_natural_sort_key)
        if not matched:
            raise FileNotFoundError(f"No files matched: {path}")
        if frame_range is not None:
            matched = matched[frame_range[0] : frame_range[1] + 1]
        frames = [cv2.imread(str(f)) for f in tqdm(matched, desc="Loading images", leave=False)]
        frames = [f for f in frames if f is not None]
        frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]

    if resize is not None and frames:
        w, h = resize
        frames = [cv2.resize(f, (w, h), interpolation=cv2.INTER_LANCZOS4) for f in frames]

    return frames


def _video_fps(path: Path) -> float | None:
    """Return fps of a video file, or the first video found in a directory."""
    vid_exts = (".mp4", ".mov", ".avi", ".mkv", ".webm")
    if path.is_dir():
        for f in sorted(path.iterdir()):
            if f.suffix.lower() in vid_exts:
                return _video_fps(f)
        return None
    if path.is_file() and path.suffix.lower() in vid_exts:
        cap = cv2.VideoCapture(str(path))
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            return float(fps) if fps > 0 else None
    return None


def _resolve_video_file(path: Path) -> Path:
    """Resolve a directory to the first video file inside it, or return path if already a file."""
    vid_exts = (".mp4", ".mov", ".avi", ".mkv", ".webm")
    if path.is_dir():
        for f in sorted(path.iterdir()):
            if f.suffix.lower() in vid_exts:
                return f
        raise FileNotFoundError(f"No video file found in directory: {path}")
    return path


def _load_video_by_timestamps(
    path: Path,
    timestamps: List[float],
    resize: Optional[Tuple[int, int]] = None,
) -> List[np.ndarray]:
    """Load frames from a video file at specific timestamps (seconds) by seeking."""
    path = _resolve_video_file(path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for t in timestamps:
        idx = min(max(0, round(t * fps)), total - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, total - 1)
            ret, frame = cap.read()
        if ret:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if resize is not None and frames:
        w, h = resize
        frames = [cv2.resize(f, (w, h), interpolation=cv2.INTER_LANCZOS4) for f in frames]
    return frames


def _get_video_frame_count(path: Path) -> int:
    """Return total frame count of a video file (or first video in directory)."""
    path = _resolve_video_file(path)
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def _load_video_by_indices(
    path: Path,
    indices: List[int],
    resize: Optional[Tuple[int, int]] = None,
) -> List[np.ndarray]:
    """Load specific frame indices from a video file."""
    path = _resolve_video_file(path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for idx in indices:
        idx = min(max(0, idx), total - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, total - 1)
            ret, frame = cap.read()
        if ret:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if resize is not None and frames:
        w, h = resize
        frames = [cv2.resize(f, (w, h), interpolation=cv2.INTER_LANCZOS4) for f in frames]
    return frames


def normalise_pair(
    original_path: str | Path,
    clean_path: str | Path,
    frame_range: Optional[Tuple[int, int]] = None,
    resize: Optional[Tuple[int, int]] = None,
) -> Tuple[List[np.ndarray], List[np.ndarray], Dict]:
    """Load and validate both plate sequences, return (orig_frames, clean_frames, meta).

    When both inputs are video files with different frame rates, the clean plate is
    loaded by timestamp (matched to the original's frame times) rather than by frame
    index, preventing temporal drift between the two plates.
    """
    op = Path(original_path)
    cp = Path(clean_path)

    orig_fps = _video_fps(op)
    clean_fps = _video_fps(cp)

    # Proportional-align when fps differs: clean[0]=orig[0] and clean[-1]=orig[-1]
    # by WAN construction, so map by relative position regardless of fps or duration.
    # Index-based loading is wrong when fps differs because clean[i] and orig[i]
    # are at different timestamps and therefore different camera positions.
    if (orig_fps is not None and clean_fps is not None
            and abs(orig_fps - clean_fps) / max(orig_fps, clean_fps) > 0.01):
        import logging
        log = logging.getLogger(__name__)
        n_orig_total = _get_video_frame_count(op)
        n_clean_total = _get_video_frame_count(cp)
        log.info("fps mismatch: orig=%.3f (%d fr) clean=%.3f (%d fr) — proportional load",
                 orig_fps, n_orig_total, clean_fps, n_clean_total)
        orig = load_frames(original_path, frame_range, resize)
        start_idx = frame_range[0] if frame_range is not None else 0
        clean_indices = [
            round((start_idx + i) * (n_clean_total - 1) / (n_orig_total - 1))
            for i in range(len(orig))
        ]
        log.info("Proportional: orig[%d..%d] -> clean[%d..%d]",
                 start_idx, start_idx + len(orig) - 1,
                 clean_indices[0], clean_indices[-1])
        clean = _load_video_by_indices(cp, clean_indices, resize)
    else:
        orig = load_frames(original_path, frame_range, resize)
        clean = load_frames(clean_path, frame_range, resize)
    if len(orig) != len(clean):
        n = min(len(orig), len(clean))
        import logging
        logging.getLogger(__name__).info(
            "Frame count mismatch after loading: orig=%d clean=%d — truncating to %d",
            len(orig), len(clean), n)
        orig, clean = orig[:n], clean[:n]

    if not orig:
        raise ValueError("No frames loaded.")
    for i, (a, b) in enumerate(zip(orig, clean)):
        if a.shape != b.shape:
            raise ValueError(
                f"Shape mismatch at frame {i}: {a.shape} vs {b.shape} — "
                "pass --resize WxH to normalise both to a common resolution")

    h, w, _ = orig[0].shape
    meta = dict(frame_count=len(orig), height=h, width=w, fps=orig_fps,
                original_path=str(original_path), clean_path=str(clean_path),
                orig_fps=orig_fps, clean_fps=clean_fps)
    return orig, clean, meta


def save_frame_sequence(frames: List[np.ndarray], out_dir: Path, prefix: str = "frame", ext: str = ".png") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx, frame in tqdm(enumerate(frames), total=len(frames), desc="Saving frames", leave=False):
        imageio.imwrite(str(out_dir / f"{prefix}{idx:04d}{ext}"), frame)


def video_from_frames(frames: List[np.ndarray], out_path: Path, fps: float = 24.0, crf: int = 23, codec: str = "libx264") -> None:
    """Write frames as MP4 via imageio-ffmpeg (temp-then-rename for safety)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=out_path.suffix, delete=False, dir=out_path.parent) as tmp:
        temp_path = Path(tmp.name)
    try:
        writer = imageio.get_writer(str(temp_path), fps=fps, format="FFMPEG",
                                     codec=codec, output_params=["-crf", str(crf)])
        for frame in tqdm(frames, desc="Writing video", leave=False):
            if frame.dtype != np.uint8:
                frame = (frame * 255).clip(0, 255).astype(np.uint8)
            writer.append_data(frame)
        writer.close()
        shutil.move(str(temp_path), str(out_path))
    except Exception as e:
        if temp_path.exists():
            temp_path.unlink()
        raise RuntimeError(f"Failed writing video: {e}") from e
