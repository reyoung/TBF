from __future__ import annotations

import os
from pathlib import Path

import torch

from tbf import TBFReader, write_tbf


def test_python_roundtrip_and_alignment(tmp_path: Path) -> None:
    out = tmp_path / "py_roundtrip.tbf"

    records = [
        {
            "a": torch.tensor([[1, 2], [3, 4]], dtype=torch.int32),
            "b": torch.tensor([1.5, -2.25], dtype=torch.float32),
        },
        {
            "c": torch.tensor([True, False, True], dtype=torch.bool),
            "d": torch.empty((0, 3), dtype=torch.float64),
        },
    ]

    write_tbf(out, records, page_size=4096)

    with TBFReader(out) as reader:
        meta = reader.metadata()
        assert len(reader) == 2
        assert reader.entry_count == 4
        for m in meta:
            assert m.data_offset % 4096 == 0

        for i in range(len(records)):
            got = reader[i]
            for key in records[i].keys():
                assert key in got
                assert got[key].dtype == records[i][key].dtype
                assert tuple(got[key].shape) == tuple(records[i][key].shape)
                assert torch.equal(got[key], records[i][key])


def test_invalid_magic(tmp_path: Path) -> None:
    path = tmp_path / "bad.tbf"
    path.write_bytes(b"not-a-tbf")

    try:
        TBFReader(path)
        assert False, "expected error"
    except ValueError:
        pass


def test_large_page_alignment(tmp_path: Path) -> None:
    out = tmp_path / "aligned_8192.tbf"
    write_tbf(out, [{"x": torch.arange(16, dtype=torch.int16)}], page_size=8192)

    with TBFReader(out) as reader:
        metas = reader.metadata()
        assert len(metas) == 1
        assert metas[0].data_offset % 8192 == 0
        assert os.path.getsize(out) > 0
