"""Tensor Batch Format constants and binary helpers."""

from __future__ import annotations

import struct
import torch

FILE_MAGIC = b"TBFDATA1"
INDEX_MAGIC = b"TBFIDX01"
FOOTER_MAGIC = b"TBFTRLR1"
VERSION = 1
DEFAULT_PAGE_SIZE = 4096

FILE_HEADER_STRUCT = struct.Struct("<8sII")
INDEX_HEADER_STRUCT = struct.Struct("<8sIQQ")
INDEX_ENTRY_PREFIX_STRUCT = struct.Struct("<QIHHQQ")
FOOTER_STRUCT = struct.Struct("<8sIQQ36s")


def align_up(value: int, alignment: int) -> int:
    if alignment <= 0:
        raise ValueError("alignment must be > 0")
    rem = value % alignment
    return value if rem == 0 else value + (alignment - rem)


def torch_dtype_maps() -> tuple[dict[object, int], dict[int, object], dict[int, int]]:
    """Return (torch_dtype->code, code->torch_dtype, code->element_size)."""
    dtype_to_code = {
        torch.float32: 1,
        torch.float64: 2,
        torch.float16: 3,
        torch.bfloat16: 4,
        torch.int8: 5,
        torch.uint8: 6,
        torch.int16: 7,
        torch.int32: 8,
        torch.int64: 9,
        torch.bool: 10,
    }
    code_to_dtype = {v: k for k, v in dtype_to_code.items()}
    code_to_elsize = {
        1: 4,
        2: 8,
        3: 2,
        4: 2,
        5: 1,
        6: 1,
        7: 2,
        8: 4,
        9: 8,
        10: 1,
    }
    return dtype_to_code, code_to_dtype, code_to_elsize
