"""Experiment runners — orchestration only, no library logic.

Each runner is a self-contained CLI that:
  - Loads weights via dms.inference.UltralyticsYOLOBackend
  - Streams a video via OpenCV
  - Calls dms.* library functions (no inline metric or policy logic)
  - Writes per-frame CSV + summary JSON to results/<stage>/<run_id>/

Run order:
  1. run_reference.py — n_only + s_only on one video (no switching)
  2. run_features.py  — per-frame proxies (no inference)
  3. run_sweep.py     — full policy sweep (only after 1 + 2 are stable)
  4. build_master_table.py — aggregate everything to master_table.csv
"""
