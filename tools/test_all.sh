#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "[test_all] syncing python deps with uv"
uv sync --dev

echo "[test_all] go tests"
(
  cd go
  go test ./...
)

echo "[test_all] c++ tests"
CPP_BIN="${ROOT_DIR}/.tmp_tbf_cpp_writer_test"
g++ -std=c++17 "$ROOT_DIR/cpp/tests/writer_test.cpp" -I"$ROOT_DIR/cpp/include" -o "$CPP_BIN"
"$CPP_BIN"
rm -f "$CPP_BIN"

echo "[test_all] python tests (includes cross-language checks)"
uv run pytest -q

echo "[test_all] all tests passed"
