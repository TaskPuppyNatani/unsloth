// EXL3-ROCm reconstruction interface. See NOTICE.md and LICENSE.exllamav3.
#pragma once

#include <ATen/Tensor.h>

void reconstruct_rocm(
    at::Tensor unpacked,
    at::Tensor packed,
    int K,
    bool mcg,
    bool mul1
);
