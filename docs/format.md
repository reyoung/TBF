# Tensor Batch Format (TBF) v1

## Goal
Store `list[dict[str, Tensor]]` in a streaming-friendly binary format with an index at the end.

## Endianness
Little-endian for all integers.

## Alignment
Tensor payload starts must be aligned to `page_size` bytes (default 4096).

## Layout
1. File header (16 bytes)
2. Tensor payload region (streamed, each tensor payload page-aligned)
3. Index region (single block written at finalize)
4. Footer (64 bytes, fixed size)

### File Header (16 bytes)
- `magic[8] = "TBFDATA1"`
- `version:u32 = 1`
- `reserved:u32 = 0`

### Index Header (28 bytes)
- `magic[8] = "TBFIDX01"`
- `version:u32 = 1`
- `entry_count:u64`
- `record_count:u64`

`entry_count` is tensor-item count, `record_count` is number of records. They are usually different when one record has multiple tensors.

### Index Entry (variable)
- `record_id:u64`
- `key_len:u32`
- `dtype_code:u16`
- `ndim:u16`
- `data_offset:u64`
- `nbytes:u64`
- `shape[i]:i64` for `i in [0, ndim)`
- `key_bytes[key_len]` UTF-8 bytes (no null terminator)

### Footer (64 bytes)
- `magic[8] = "TBFTRLR1"`
- `version:u32 = 1`
- `index_offset:u64`
- `index_size:u64`
- `reserved[36] = 0`

## DType codes
- `1`: `float32`
- `2`: `float64`
- `3`: `float16`
- `4`: `bfloat16`
- `5`: `int8`
- `6`: `uint8`
- `7`: `int16`
- `8`: `int32`
- `9`: `int64`
- `10`: `bool`

## Streaming write model
- Records are appended in order: record 0, record 1, ...
- Within a record, tensors are written in dict iteration order.
- Payload bytes are written immediately when each tensor arrives.
- Index is buffered in memory and written once at close.

## Reader reconstruction
- Parse footer from file end.
- Parse index.
- Rebuild `list[dict[str, Tensor]]` by `record_id` and `key`.
