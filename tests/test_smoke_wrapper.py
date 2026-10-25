"""Offline PlateLock smoke wrapper contract tests (no models or GPU)."""
from __future__ import annotations
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/smoke_test.sh"

FAKE_PY = r"""#!/usr/bin/env bash
if [ "$#" -eq 0 ]; then exit "$FAKE_FAIL_INPUT"; fi
if [ "$1" = "-c" ]; then exit "$FAKE_FAIL_ENV"; fi
if [ "$1" = "-" ]; then exec /usr/bin/python3 "$@"; fi
case "$1" in
  */scripts/run_method.py)
    shift
    method=""
    result_dir=""
    while [ "$#" -gt 0 ]; do
      case "$1" in
        --method) method="$2"; shift 2 ;;
        --out) result_dir="$2"; shift 2 ;;
        *) shift ;;
      esac
    done
    if [ "$method" = "$FAKE_FAIL_METHOD" ]; then exit 25; fi
    mkdir -p "$result_dir"
    if [ "$method" != "$FAKE_SKIP_RESULT_METHOD" ]; then
      /usr/bin/python3 -c \
        'import json,sys; json.dump({"method":sys.argv[2],"frame_count":int(sys.argv[3]),"metrics_summary":{"mean_psnr":1,"mean_ssim":1,"mean_mae":1}},open(sys.argv[1],"w"))' \
        "$result_dir/method_result.json" "$method" "$FAKE_COUNT"
    fi
    exit 0
    ;;
esac
exit 2
"""


class OfflineSmokeWrapperTests(unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory(prefix="platelock_smoke_wrapper_")
        self.addCleanup(self.scratch.cleanup)
        base = Path(self.scratch.name)
        self.repo = base / "repo"
        (self.repo / "scripts").mkdir(parents=True)
        (self.repo / "configs").mkdir()
        (self.repo / "configs/debug_5frames.yaml").write_text("mode: debug\n")
        shutil.copy2(SCRIPT, self.repo / "scripts/smoke_test.sh")
        fakebin = base / "fakebin"
        fakebin.mkdir()
        stub = fakebin / "python3"
        stub.write_text(FAKE_PY)
        stub.chmod(0o755)
        self.env = {**os.environ, "PATH": str(fakebin) + os.pathsep + os.environ["PATH"],
                    "FAKE_FAIL_ENV": "0", "FAKE_FAIL_INPUT": "0",
                    "FAKE_FAIL_METHOD": "never", "FAKE_SKIP_RESULT_METHOD": "never",
                    "FAKE_COUNT": "5"}

    def call(self, **extras):
        return subprocess.run(["bash", str(self.repo / "scripts/smoke_test.sh")],
                              cwd=self.repo, env={**self.env, **extras},
                              capture_output=True, text=True, timeout=15)

    def test_all_methods_pass_only_with_valid_metadata(self):
        p = self.call()
        self.assertEqual(p.returncode, 0, p.stderr + p.stdout)
        for method in ("ECC", "SEARAFT", "COTRACKER_MESH"):
            self.assertIn(method + ": PASS", p.stdout)
        self.assertIn("smoke_test: PASS", p.stdout)

    def test_method_nonzero_propagates_to_overall_status(self):
        p = self.call(FAKE_FAIL_METHOD="searaft")
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("SEARAFT: FAIL", p.stdout)
        self.assertIn("smoke_test: FAIL", p.stdout)

    def test_missing_method_manifest_is_failure(self):
        p = self.call(FAKE_SKIP_RESULT_METHOD="ecc")
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("ECC: FAIL", p.stdout)

    def test_wrong_frame_count_is_failure(self):
        p = self.call(FAKE_COUNT="4")
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("COTRACKER_MESH: FAIL", p.stdout)

    def test_torch_preflight_failure_aborts(self):
        p = self.call(FAKE_FAIL_ENV="1")
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("ENV: FAIL", p.stdout)
        self.assertNotIn("ECC: PASS", p.stdout)

    def test_input_generation_failure_aborts(self):
        p = self.call(FAKE_FAIL_INPUT="1")
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("INPUT: FAIL", p.stdout)
        self.assertNotIn("ECC: PASS", p.stdout)

    def test_successive_runs_use_unique_directories(self):
        self.assertEqual(self.call().returncode, 0)
        self.assertEqual(self.call().returncode, 0)
        results = list((self.repo / "output/smoke_tests").glob("run_*"))
        self.assertEqual(len(results), 2)


if __name__ == "__main__":
    unittest.main()
