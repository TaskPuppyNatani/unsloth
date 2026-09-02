// Adapted from ExLlamaV3 quant/codebook.cuh and quant/exl3_dq.cuh.
// Copyright (c) 2025 Turboderp. See LICENSE.exllamav3 and NOTICE.md.
#pragma once

#include <cstdint>

#include <hip/hip_fp16.h>

#include "portable_codebook.h"

namespace exl3_rocm
{

template <int Codebook>
__device__ __forceinline__
__half decode_3inst_device(
    std::uint32_t x
)
{
    static_assert(
        Codebook >= 0 && Codebook <= 2,
        "EXL3 codebook must be 0, 1, or 2"
    );

    if constexpr (
        Codebook == 0 ||
        Codebook == 1
    )
    {
        const std::uint32_t word =
            codebook_transform<Codebook>(x);

        const __half lo =
            __ushort_as_half(
                static_cast<std::uint16_t>(
                    word & 0xffffu
                )
            );

        const __half hi =
            __ushort_as_half(
                static_cast<std::uint16_t>(
                    word >> 16
                )
            );

        return __hadd(lo, hi);
    }
    else
    {
        const std::uint16_t sum_bits =
            mul1_accumulator_bits(x);

        const __half h =
            __ushort_as_half(sum_bits);

        const __half k_inv =
            __ushort_as_half(0x1eeeu);

        const __half k_bias =
            __ushort_as_half(0xc931u);

        return __hfma(
            h,
            k_inv,
            k_bias
        );
    }
}


template <int Bits>
__device__ __forceinline__
std::uint16_t trellis_window_device(
    const std::uint32_t* ptr,
    int t_offset
)
{
    static_assert(
        Bits >= 1 && Bits <= 8,
        "EXL3 bitrate must be 1 through 8"
    );

    constexpr int total_bits =
        Bits * 256;

    constexpr int total_words =
        total_bits / 32;

    const int b0 =
        t_offset * Bits +
        Bits -
        16 +
        total_bits;

    const int b1 =
        b0 + 16;

    const int i0 =
        b0 / 32;

    const int i1 =
        (b1 - 1) / 32;

    const int shift =
        (i1 + 1) * 32 -
        b1;

    const std::uint32_t a =
        ptr[i0 % total_words];

    const std::uint32_t b =
        ptr[i1 % total_words];

    return static_cast<std::uint16_t>(
        concat_shift_right(
            b,
            a,
            static_cast<unsigned int>(
                shift
            )
        ) &
        0xffffu
    );
}


template <int Bits, int Codebook>
__device__ __forceinline__
__half dq_device(
    const std::uint32_t* ptr,
    int t_offset
)
{
    return decode_3inst_device<
        Codebook
    >(
        trellis_window_device<
            Bits
        >(
            ptr,
            t_offset
        )
    );
}

} // namespace exl3_rocm
