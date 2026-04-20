#!/usr/bin/env bash
# Build native extensions for a local conda environment.
# BundleSDF CUDA ops (bundlesdf/mycuda) are omitted; model-free NeRF path needs them separately.
set -euo pipefail

PROJ_ROOT="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

: "${CONDA_PREFIX:?Set CONDA_PREFIX by activating your conda env first.}"

export CMAKE_PREFIX_PATH="${CONDA_PREFIX}:${CMAKE_PREFIX_PATH:-}"

cd "${PROJ_ROOT}/mycpp"
rm -rf build
mkdir -p build
cd build
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH}" \
  -DPython3_ROOT_DIR="${CONDA_PREFIX}" \
  -DPYBIND11_PYTHON_EXECUTABLE="${CONDA_PREFIX}/bin/python"
cmake --build . -j"$(nproc)"

cd "${PROJ_ROOT}"
