#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "tbf_writer.hpp"

static std::vector<uint8_t> ToBytesI64(const std::vector<int64_t>& v) {
  std::vector<uint8_t> out(v.size() * sizeof(int64_t));
  std::memcpy(out.data(), v.data(), out.size());
  return out;
}

static std::vector<uint8_t> ToBytesF32(const std::vector<float>& v) {
  std::vector<uint8_t> out(v.size() * sizeof(float));
  std::memcpy(out.data(), v.data(), out.size());
  return out;
}

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: " << argv[0] << " <output.tbf>\n";
    return 2;
  }

  tbf::Writer writer(argv[1]);

  {
    std::vector<int64_t> a = {11, 22, 33};
    std::vector<float> b = {3.0f, -4.5f};
    writer.add_record({
        tbf::TensorItem{"a", tbf::DType::Int64, {3}, ToBytesI64(a)},
        tbf::TensorItem{"b", tbf::DType::Float32, {2}, ToBytesF32(b)},
    });
  }

  {
    std::vector<int64_t> c = {100, 200};
    writer.add_record({tbf::TensorItem{"c", tbf::DType::Int64, {2}, ToBytesI64(c)}});
  }

  writer.close();
  return 0;
}
