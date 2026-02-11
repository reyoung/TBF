# TBF (Tensor Batch Format)

TBF is a binary format for storing `list[dict[str, Tensor]]` with:
- mmap-friendly tensor payload layout (each tensor payload is page-aligned)
- streaming writes
- index-at-end design (RecordIO-like)

## Language support

- Python: read + write (only dependency is PyTorch)
- Go: write
- C++: write (header-only)

## Format summary

- File header (`TBFDATA1`)
- Payload region (streamed tensor bytes, each tensor payload page-aligned)
- Index region at file end (`TBFIDX01`)
- Footer (`TBFTRLR1`) containing index location/size

Specification: `docs/format.md`

## Python usage

```python
from tbf import TBFWriter, TBFReader
import torch

records = [
    {"x": torch.tensor([1, 2, 3], dtype=torch.int32)},
    {"y": torch.tensor([[1.0, 2.0]], dtype=torch.float32)},
]

with TBFWriter("sample.tbf", page_size=4096) as w:
    for r in records:
        w.add_record(r)

with TBFReader("sample.tbf") as r:
    n = len(r)
    first = r[0]
    print(n, first["x"])
```

## Go usage (writer)

```go
w, err := tbf.NewWriter("sample.tbf", tbf.DefaultPageSize)
if err != nil { panic(err) }
defer w.Close()

item := tbf.TensorItem{
    Key:   "x",
    DType: tbf.Int32,
    Shape: []int64{3},
    Data:  []byte{1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0},
}
if err := w.AddRecord([]tbf.TensorItem{item}); err != nil {
    panic(err)
}
```

## C++ usage (header-only writer)

```cpp
#include "tbf_writer.hpp"

tbf::Writer writer("sample.tbf");
std::vector<int64_t> values = {1, 2, 3};
std::vector<uint8_t> data(values.size() * sizeof(int64_t));
std::memcpy(data.data(), values.data(), data.size());

writer.add_record({tbf::TensorItem{"x", tbf::DType::Int64, {3}, data}});
writer.close();
```

## Development and tests

Python dependencies are managed with `uv`.

Run all tests (Go + C++ + Python + cross-language):

```bash
tools/test_all.sh
```

This script will:
- `uv sync --dev`
- run `go test ./...`
- compile and run C++ tests
- run `uv run pytest -q`
