from __future__ import annotations

import subprocess
from pathlib import Path

import torch

from tbf import TBFReader


ROOT = Path(__file__).resolve().parents[2]


def run(cmd: list[str], cwd: Path | None = None) -> None:
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def test_go_writer_compatible_with_python_reader(tmp_path: Path) -> None:
    out = tmp_path / "go_fixture.tbf"
    run(["go", "run", "./tests/gen_fixture", str(out)], cwd=ROOT / "go")

    with TBFReader(out) as reader:
        assert len(reader) == 2
        assert reader.entry_count == 3
        first = reader[0]
        second = reader[1]

    assert torch.equal(first["x"], torch.tensor([[1, 2], [3, 4]], dtype=torch.int32))
    assert torch.equal(first["y"], torch.tensor([9, 8, 7], dtype=torch.uint8))
    assert torch.allclose(second["z"], torch.tensor([1.5, -2.0], dtype=torch.float32))


def test_cpp_writer_compatible_with_python_reader(tmp_path: Path) -> None:
    bin_path = tmp_path / "cpp_fixture"
    out = tmp_path / "cpp_fixture.tbf"

    run([
        "g++",
        "-std=c++17",
        str(ROOT / "cpp/tests/write_fixture.cpp"),
        "-I",
        str(ROOT / "cpp/include"),
        "-o",
        str(bin_path),
    ])
    run([str(bin_path), str(out)])

    with TBFReader(out) as reader:
        assert len(reader) == 2
        assert reader.entry_count == 3
        first = reader[0]
        second = reader[1]

    assert torch.equal(first["a"], torch.tensor([11, 22, 33], dtype=torch.int64))
    assert torch.allclose(first["b"], torch.tensor([3.0, -4.5], dtype=torch.float32))
    assert torch.equal(second["c"], torch.tensor([100, 200], dtype=torch.int64))
