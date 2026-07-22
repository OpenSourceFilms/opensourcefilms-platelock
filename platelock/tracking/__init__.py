"""platelock.tracking — same-domain multi-keyframe tracking pipeline.

Alternative to the RoMa-dense cross-domain-matching branch (platelock.dense_field
+ platelock.run_full_sequence): tracks each plate independently with CoTracker3
and bootstraps keyframe correspondence via an already-solved backward map,
rather than re-matching original<->clean every frame. Validated end-to-end on
the ExampleShot shot (recipe name during development: "multikey13").

Both branches converge on the same assembly unit (platelock.dense_field.M/cres
+ assemble_map), so downstream code (export, diagnostics) doesn't care which
correspondence source produced a given frame's field.
"""
