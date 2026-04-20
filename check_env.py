#!/usr/bin/env python3
"""Quick sanity checks for a local FoundationPose install (no inference)."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--demo-data",
    action="store_true",
    help="Also check default paths used by run_demo.py (demo_data/mustard0).",
  )
  args = parser.parse_args()

  repo_root = Path(__file__).resolve().parent
  os.chdir(repo_root)
  if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

  errors: list[str] = []
  warnings: list[str] = []

  print(f"Repository: {repo_root}")
  print(f"Python: {sys.version.split()[0]}")

  # --- PyTorch / CUDA ---
  try:
    import torch

    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
      print(f"CUDA device: {torch.cuda.get_device_name(0)}")
  except Exception as e:
    errors.append(f"torch: {e}")

  # --- Core imports (same chain as run_demo / estimater) ---
  modules = [
    "pytorch3d",
    "nvdiffrast.torch",
    "trimesh",
    "open3d",
    "cv2",
    "warp",
  ]
  for mod in modules:
    try:
      __import__(mod)
      print(f"import {mod}: ok")
    except Exception as e:
      errors.append(f"import {mod}: {e}")

  # --- Native extension ---
  build_dir = repo_root / "mycpp" / "build"
  so_files = list(build_dir.glob("mycpp*.so"))
  if not so_files:
    warnings.append(f"No mycpp extension found under {build_dir} (run bash build_all_conda.sh).")
  else:
    print(f"mycpp extension: {so_files[0].name}")

  try:
    sys.path.insert(0, str(build_dir))
    import mycpp

    print(f"import mycpp: ok ({mycpp})")
  except Exception as e:
    errors.append(f"import mycpp: {e}")

  try:
    import estimater

    print("import estimater: ok")
  except Exception as e:
    errors.append(f"import estimater: {e}")

  # --- Weights (hard-coded run names in learning/training/predict_*.py) ---
  weight_sets = [
    ("scorer", "2024-01-11-20-02-45"),
    ("refiner", "2023-10-28-18-33-37"),
  ]
  weights_root = repo_root / "weights"
  for label, stamp in weight_sets:
    d = weights_root / stamp
    ckpt = d / "model_best.pth"
    cfg = d / "config.yml"
    missing = [p.name for p in (ckpt, cfg) if not p.is_file()]
    if missing:
      warnings.append(
        f"Weights ({label}, {stamp}): missing {', '.join(missing)} — download per readme Data prepare."
      )
    else:
      print(f"Weights ({label}, {stamp}): ok")

  # --- Optional demo bundle ---
  if args.demo_data:
    demo_scene = repo_root / "demo_data" / "mustard0"
    demo_mesh = demo_scene / "mesh" / "textured_simple.obj"
    if not demo_mesh.is_file():
      warnings.append(f"run_demo demo scene: missing {demo_mesh}")
    else:
      print(f"run_demo demo_data: ok ({demo_mesh.relative_to(repo_root)})")

  print()
  for w in warnings:
    print(f"WARNING: {w}")
  for e in errors:
    print(f"ERROR: {e}")

  # Non-zero only when imports or core checks fail; missing weights/data still exit 0.
  if errors:
    return 1
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
