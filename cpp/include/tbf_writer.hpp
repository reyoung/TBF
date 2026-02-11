#pragma once

#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace tbf {

static constexpr uint32_t kVersion = 1;
static constexpr uint32_t kDefaultPageSize = 4096;

enum class DType : uint16_t {
  Float32 = 1,
  Float64 = 2,
  Float16 = 3,
  BFloat16 = 4,
  Int8 = 5,
  UInt8 = 6,
  Int16 = 7,
  Int32 = 8,
  Int64 = 9,
  Bool = 10,
};

inline uint64_t dtype_element_size(DType dtype) {
  switch (dtype) {
    case DType::Float32:
      return 4;
    case DType::Float64:
      return 8;
    case DType::Float16:
    case DType::BFloat16:
    case DType::Int16:
      return 2;
    case DType::Int8:
    case DType::UInt8:
    case DType::Bool:
      return 1;
    case DType::Int32:
      return 4;
    case DType::Int64:
      return 8;
  }
  throw std::invalid_argument("unsupported dtype");
}

struct TensorItem {
  std::string key;
  DType dtype;
  std::vector<int64_t> shape;
  std::vector<uint8_t> data;
};

class Writer {
 public:
  explicit Writer(const std::string& path, uint32_t page_size = kDefaultPageSize)
      : page_size_(page_size), record_count_(0), closed_(false) {
    if (page_size_ == 0) {
      throw std::invalid_argument("page_size must be > 0");
    }
    out_.open(path, std::ios::binary | std::ios::trunc);
    if (!out_) {
      throw std::runtime_error("failed to open output file");
    }
    write_file_header();
  }

  ~Writer() {
    try {
      close();
    } catch (...) {
    }
  }

  void add_record(const std::vector<TensorItem>& items) {
    const uint64_t record_id = record_count_;
    for (const auto& item : items) {
      add_tensor(record_id, item);
    }
    record_count_++;
  }

  void add_tensor(uint64_t record_id, const TensorItem& item) {
    if (closed_) {
      throw std::runtime_error("writer already closed");
    }
    if (item.key.empty()) {
      throw std::invalid_argument("tensor key cannot be empty");
    }
    const uint64_t expected = expected_nbytes(item.shape, dtype_element_size(item.dtype));
    if (item.data.size() != expected) {
      throw std::invalid_argument("tensor data size mismatch");
    }

    const uint64_t cur = static_cast<uint64_t>(out_.tellp());
    const uint64_t aligned = align_up(cur, page_size_);
    const uint64_t pad = aligned - cur;
    if (pad > 0) {
      const std::vector<uint8_t> zeros(static_cast<size_t>(pad), 0);
      out_.write(reinterpret_cast<const char*>(zeros.data()), static_cast<std::streamsize>(zeros.size()));
    }

    if (!item.data.empty()) {
      out_.write(reinterpret_cast<const char*>(item.data.data()), static_cast<std::streamsize>(item.data.size()));
    }

    entries_.push_back(IndexEntry{record_id, item.key, static_cast<uint16_t>(item.dtype), item.shape, aligned,
                                  static_cast<uint64_t>(item.data.size())});
  }

  void close() {
    if (closed_) {
      return;
    }
    write_index_and_footer();
    out_.close();
    closed_ = true;
  }

 private:
  struct IndexEntry {
    uint64_t record_id;
    std::string key;
    uint16_t dtype_code;
    std::vector<int64_t> shape;
    uint64_t data_offset;
    uint64_t nbytes;
  };

  std::ofstream out_;
  uint32_t page_size_;
  uint64_t record_count_;
  std::vector<IndexEntry> entries_;
  bool closed_;

  static uint64_t align_up(uint64_t value, uint64_t alignment) {
    if (alignment == 0 || value % alignment == 0) {
      return value;
    }
    return value + (alignment - (value % alignment));
  }

  static uint64_t expected_nbytes(const std::vector<int64_t>& shape, uint64_t element_size) {
    if (shape.empty()) {
      return element_size;
    }
    uint64_t prod = 1;
    for (int64_t d : shape) {
      if (d < 0) {
        throw std::invalid_argument("negative shape dim");
      }
      prod *= static_cast<uint64_t>(d);
    }
    return prod * element_size;
  }

  template <typename T>
  void write_le(T value) {
    for (size_t i = 0; i < sizeof(T); ++i) {
      const uint8_t byte = static_cast<uint8_t>((static_cast<uint64_t>(value) >> (8 * i)) & 0xff);
      out_.put(static_cast<char>(byte));
    }
  }

  void write_i64_le(int64_t value) {
    const uint64_t u = static_cast<uint64_t>(value);
    for (size_t i = 0; i < sizeof(uint64_t); ++i) {
      const uint8_t byte = static_cast<uint8_t>((u >> (8 * i)) & 0xff);
      out_.put(static_cast<char>(byte));
    }
  }

  void write_file_header() {
    static const char magic[8] = {'T', 'B', 'F', 'D', 'A', 'T', 'A', '1'};
    out_.write(magic, 8);
    write_le<uint32_t>(kVersion);
    write_le<uint32_t>(0);
  }

  void write_index_and_footer() {
    const uint64_t index_offset = static_cast<uint64_t>(out_.tellp());

    static const char index_magic[8] = {'T', 'B', 'F', 'I', 'D', 'X', '0', '1'};
    out_.write(index_magic, 8);
    write_le<uint32_t>(kVersion);
    write_le<uint64_t>(static_cast<uint64_t>(entries_.size()));
    write_le<uint64_t>(record_count_);

    for (const auto& e : entries_) {
      write_le<uint64_t>(e.record_id);
      write_le<uint32_t>(static_cast<uint32_t>(e.key.size()));
      write_le<uint16_t>(e.dtype_code);
      write_le<uint16_t>(static_cast<uint16_t>(e.shape.size()));
      write_le<uint64_t>(e.data_offset);
      write_le<uint64_t>(e.nbytes);
      for (int64_t dim : e.shape) {
        write_i64_le(dim);
      }
      out_.write(e.key.data(), static_cast<std::streamsize>(e.key.size()));
    }

    const uint64_t index_size = static_cast<uint64_t>(out_.tellp()) - index_offset;

    static const char footer_magic[8] = {'T', 'B', 'F', 'T', 'R', 'L', 'R', '1'};
    out_.write(footer_magic, 8);
    write_le<uint32_t>(kVersion);
    write_le<uint64_t>(index_offset);
    write_le<uint64_t>(index_size);
    for (int i = 0; i < 36; ++i) {
      out_.put('\0');
    }
  }
};

}  // namespace tbf
