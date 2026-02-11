#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "tbf_writer.hpp"

namespace {

uint16_t ReadU16LE(const std::vector<uint8_t>& buf, size_t pos) {
  return static_cast<uint16_t>(buf[pos]) | (static_cast<uint16_t>(buf[pos + 1]) << 8);
}

uint32_t ReadU32LE(const std::vector<uint8_t>& buf, size_t pos) {
  uint32_t v = 0;
  for (int i = 0; i < 4; ++i) {
    v |= static_cast<uint32_t>(buf[pos + i]) << (8 * i);
  }
  return v;
}

uint64_t ReadU64LE(const std::vector<uint8_t>& buf, size_t pos) {
  uint64_t v = 0;
  for (int i = 0; i < 8; ++i) {
    v |= static_cast<uint64_t>(buf[pos + i]) << (8 * i);
  }
  return v;
}

std::vector<uint8_t> ReadFile(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  assert(in.good());
  return std::vector<uint8_t>(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

std::vector<uint8_t> ToBytesI32(const std::vector<int32_t>& v) {
  std::vector<uint8_t> out(v.size() * sizeof(int32_t));
  std::memcpy(out.data(), v.data(), out.size());
  return out;
}

}  // namespace

int main() {
  const auto out_path = (std::filesystem::temp_directory_path() / "tbf_cpp_writer_test.tbf").string();

  {
    tbf::Writer writer(out_path, 4096);

    std::vector<int32_t> a = {1, 2, 3, 4};
    writer.add_record({tbf::TensorItem{"a", tbf::DType::Int32, {2, 2}, ToBytesI32(a)}});

    std::vector<int32_t> b = {5, 6};
    writer.add_record({tbf::TensorItem{"b", tbf::DType::Int32, {2}, ToBytesI32(b)}});

    writer.close();
  }

  const auto buf = ReadFile(out_path);
  assert(buf.size() > 80);

  assert(std::memcmp(buf.data(), "TBFDATA1", 8) == 0);
  assert(ReadU32LE(buf, 8) == 1);

  const size_t footer_pos = buf.size() - 64;
  assert(std::memcmp(buf.data() + footer_pos, "TBFTRLR1", 8) == 0);
  assert(ReadU32LE(buf, footer_pos + 8) == 1);
  const uint64_t index_offset = ReadU64LE(buf, footer_pos + 12);
  const uint64_t index_size = ReadU64LE(buf, footer_pos + 20);

  assert(index_offset + index_size <= footer_pos);

  assert(std::memcmp(buf.data() + index_offset, "TBFIDX01", 8) == 0);
  assert(ReadU32LE(buf, index_offset + 8) == 1);
  const uint64_t entry_count = ReadU64LE(buf, index_offset + 12);
  const uint64_t record_count = ReadU64LE(buf, index_offset + 20);
  assert(entry_count == 2);
  assert(record_count == 2);

  size_t pos = static_cast<size_t>(index_offset + 28);
  for (uint64_t i = 0; i < entry_count; ++i) {
    const uint64_t record_id = ReadU64LE(buf, pos);
    const uint32_t key_len = ReadU32LE(buf, pos + 8);
    const uint16_t ndim = ReadU16LE(buf, pos + 14);
    const uint64_t data_offset = ReadU64LE(buf, pos + 16);

    assert(record_id == i);
    assert(data_offset % 4096 == 0);

    pos += 32;
    pos += static_cast<size_t>(ndim) * 8;
    pos += key_len;
  }

  std::remove(out_path.c_str());
  return 0;
}
