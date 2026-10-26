# Reproduce the synthetic PlateLock before/after

![Three-panel synthetic registration schematic](registration_synthetic.svg)

The SVG above is an **illustrative schematic** of a known translation;
it is **not** a frame extracted from a production shot and it is not pixel
output of a trained tracker. For actual rendered pixels, use the script below.
It calls the **real public PlateLock** backward-map implementation.

## Generate an actual three-panel PNG

Requires Python 3.11+, NumPy and OpenCV (no torch, model weights or GPU):

~~~bash
python3 -m pip install "numpy==2.2.6" "opencv-python-headless==4.11.0.86"
python3 -B examples/synthetic_registration_demo.py --out /tmp/platelock-registration-demo.png
~~~

The input is a made-up geometric illustration generated in memory:
no external image, original footage or production media is read.
The demonstration creates three panels:

1. Synthetic original (the target coordinate system).
2. Deliberately shifted copy, called a synthetic "clean" plate solely for
   demonstrating the registration transform.
3. The result of applying a **known** output-to-source backward map using
   platelock.geometry.affine_to_backward_map() and apply_backward_map().

The known translation is (+16 pixels X, -10 pixels Y). The script checks
that mean absolute BGR pixel error in a 24-pixel interior crop decreases
after mapping and is zero for this integer translation. The crop removes
edge regions where pixels have no corresponding source due to shifting.
A sample local run produced approximately 66.61 MAE before correction and
0 after correction. This is a deliberately easy registration sanity check,
not a realistic occlusion/parallax/temporal evaluation.

The script refuses to overwrite an existing PNG. Choose a new output
filename on each run, or intentionally remove only a disposable prior
demo file yourself. It does not modify PlateLock configuration or touch
your normal input/output directories.

## Acceptance and limits

~~~bash
python3 -B -m unittest discover -s tests -p 'test_synthetic_registration_demo.py' -v
~~~

Public CI tests the actual PNG, output dimensions, registration accuracy,
determinism, CLI execution and non-overwrite behavior with no neural models.
It is **not** proof that CoTracker, SEA-RAFT, SAM 3, STMap export, fine
registration on production plates or Nuke integration all work on a
fresh GPU. Those require separate real integration and shot acceptance.
