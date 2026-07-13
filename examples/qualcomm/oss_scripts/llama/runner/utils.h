/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>

// Widen an IEEE binary16 bit pattern. __fp16 is an ARM extension and does not
// exist under gcc/x86_64, which is how the host leg of build.sh compiles this.
inline float fp16_to_fp32(uint16_t h) {
  uint32_t sign = (h >> 15) & 0x1u;
  uint32_t exp = (h >> 10) & 0x1fu;
  uint32_t mant = h & 0x3ffu;
  uint32_t f;
  if (exp == 0) {
    if (mant == 0) {
      f = sign << 31;
    } else {
      exp = 1;
      while (!(mant & 0x400)) {
        mant <<= 1;
        exp--;
      }
      mant &= 0x3ff;
      f = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
    }
  } else if (exp == 31) {
    f = (sign << 31) | 0x7f800000u | (mant << 13);
  } else {
    f = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
  }
  float result;
  std::memcpy(&result, &f, sizeof(result));
  return result;
}

// Template struct to hold tensor data and tensor

// TODO: Refactor these struct to use TensorPtr
// see https://docs.pytorch.org/executorch/stable/extension-tensor.html

// TensorStruct whose dtype known in compile time
template <typename T>
struct TensorStruct {
  std::unique_ptr<executorch::aten::TensorImpl> tensor;
  std::shared_ptr<std::vector<T>> buffer;
  T* data;
  // data size in bytes
  size_t size;
};

inline size_t getDtypeSize(executorch::aten::ScalarType dtype) {
  switch (dtype) {
    case executorch::aten::ScalarType::Float:
      return sizeof(float);
    case executorch::aten::ScalarType::Double:
      return sizeof(double);
    case executorch::aten::ScalarType::Int:
      return sizeof(int32_t);
    case executorch::aten::ScalarType::Long:
      return sizeof(int64_t);
    case executorch::aten::ScalarType::Byte:
      return sizeof(uint8_t);
    case executorch::aten::ScalarType::UInt16:
      return sizeof(uint16_t);
    case executorch::aten::ScalarType::Half:
      return sizeof(uint16_t);
    default:
      ET_CHECK_MSG(
          false,
          "Unsupported scalar type %s",
          executorch::runtime::toString(dtype));
      break;
  }
}

// TensorStruct whose dtype known in runtime, and raw file is used
struct TensorStructRaw {
  std::unique_ptr<executorch::aten::TensorImpl> tensor;
  std::byte* data;
  // data size in bytes
  size_t size;
  executorch::aten::ScalarType dtype;
  size_t getElementSize() const {
    return getDtypeSize(dtype);
  }
};
