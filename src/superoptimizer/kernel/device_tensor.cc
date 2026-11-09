#include "cleave/superoptimizer/kernel/device_tensor.h"
#include "cleave/superoptimizer/utils/hash_utils.h"
#include "cleave/superoptimizer/utils/logging.h"
#include <functional>
#include <stdexcept>

namespace cleave_superoptimizer {
namespace type {

namespace {
struct DataTypeName {
  DataType dtype;
  char const *name;
};
constexpr DataTypeName kDataTypeNames[] = {
    {DT_INT8, "int8"},       {DT_INT16, "int16"},
    {DT_INT32, "int32"},     {DT_INT64, "int64"},
    {DT_UINT8, "uint8"},     {DT_UINT16, "uint16"},
    {DT_UINT32, "uint32"},   {DT_UINT64, "uint64"},
    {DT_FLOAT16, "float16"}, {DT_BFLOAT16, "bfloat16"},
    {DT_FLOAT32, "float32"}, {DT_FLOAT64, "float64"},
};
} // namespace

DataType data_type_from_string(std::string const &name) {
  if (name.empty()) {
    return DT_UNKNOWN;
  }
  for (auto const &entry : kDataTypeNames) {
    if (name == entry.name) {
      return entry.dtype;
    }
  }
  CSO_THROW(std::invalid_argument, "Unknown dtype '" << name << "'");
}

std::string data_type_to_string(DataType dtype) {
  for (auto const &entry : kDataTypeNames) {
    if (dtype == entry.dtype) {
      return entry.name;
    }
  }
  return "";
}

} // namespace type

namespace kernel {

DTensor::DTensor() {
  num_dims = 0;
  for (int i = 0; i < MAX_TENSOR_DIMS; i++) {
    dim[i] = 0;
    can_reduce[i] = false;
    can_split[i] = false;
  }
  owner_op = nullptr;
  owner_ts_idx = -1000;
  dtype = type::DT_UNKNOWN;
}

std::atomic<int64_t> DTensor::next_guid = 10000000;

void to_json(json &j, DTensor const &t) {
  j = json{{"num_dims", t.num_dims},
           {"dim", t.dim},
           {"can_reduce", t.can_reduce},
           {"can_split", t.can_split},
           {"guid", t.guid}};
  if (t.dtype != type::DT_UNKNOWN) {
    j["dtype"] = type::data_type_to_string(t.dtype);
  }
}

void from_json(json const &j, DTensor &t) {
  j.at("num_dims").get_to(t.num_dims);
  j.at("dim").get_to(t.dim);
  j.at("can_reduce").get_to(t.can_reduce);
  j.at("can_split").get_to(t.can_split);
  j.at("guid").get_to(t.guid);
  t.dtype = type::data_type_from_string(j.value("dtype", std::string()));
}

} // namespace kernel
} // namespace cleave_superoptimizer

namespace std {

size_t hash<cleave_superoptimizer::kernel::DTensor>::operator()(
    cleave_superoptimizer::kernel::DTensor const &tensor) const {
  size_t ret = hash<int>()(tensor.num_dims);
  for (int i = 0; i < tensor.num_dims; i++) {
    hash_combine(ret, tensor.dim[i]);
    hash_combine(ret, tensor.can_reduce[i]);
    hash_combine(ret, tensor.can_split[i]);
  }
  hash_combine(ret, tensor.owner_op);
  hash_combine(ret, tensor.owner_ts_idx);
  return ret;
}

} // namespace std
