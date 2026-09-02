// Adapted from ExLlamaV3 quant/codebook.cuh. Copyright (c) 2025 Turboderp.
// See LICENSE.exllamav3 and NOTICE.md for the MIT notice and source provenance.
#pragma once

#include <cstdint>

#include "portable_bitops.h"

#if defined(__CUDACC__) || defined(__HIPCC__)
#define EXL3_CB_HD __host__ __device__ __forceinline__
#else
#define EXL3_CB_HD inline
#endif

namespace exl3_rocm
{

template <int Codebook>
EXL3_CB_HD std::uint32_t codebook_transform(std::uint32_t x)
{
    static_assert(
        Codebook == 0 || Codebook == 1,
        "codebook_transform supports codebooks 0 and 1"
    );

    if constexpr (Codebook == 0)
    {
        x *= 89226354u;
        x += 64248484u;
    }
    else
    {
        x *= 0xCBAC1FEDu;
    }

    return lop3<0x6a>(
        x,
        0x8fff8fffu,
        0x3b603b60u
    );
}

// Codebook 2 ("mul1") takes a different path upstream.
// This returns the 16-bit value that is reinterpreted as FP16
// immediately before the final half-precision FMA.
EXL3_CB_HD std::uint16_t mul1_accumulator_bits(std::uint32_t x)
{
    x *= 0x83DCD12Du;

    return static_cast<std::uint16_t>(
        dp4a_ones_u8(x, 0x6400u)
    );
}

} // namespace exl3_rocm

#undef EXL3_CB_HD
