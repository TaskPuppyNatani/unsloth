// EXL3-ROCm portable bit operations. See NOTICE.md for provenance and
// LICENSE.exllamav3 for the bundled upstream permission notice.
#pragma once

#include <cstdint>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define EXL3_HD __host__ __device__ __forceinline__
#else
#define EXL3_HD inline
#endif

namespace exl3_rocm
{

template <std::uint8_t Lut>
EXL3_HD std::uint32_t lop3(
    std::uint32_t a,
    std::uint32_t b,
    std::uint32_t c
)
{
    std::uint32_t r = 0;

    if constexpr (Lut & 0x01) r |= (~a & ~b & ~c);
    if constexpr (Lut & 0x02) r |= (~a & ~b &  c);
    if constexpr (Lut & 0x04) r |= (~a &  b & ~c);
    if constexpr (Lut & 0x08) r |= (~a &  b &  c);
    if constexpr (Lut & 0x10) r |= ( a & ~b & ~c);
    if constexpr (Lut & 0x20) r |= ( a & ~b &  c);
    if constexpr (Lut & 0x40) r |= ( a &  b & ~c);
    if constexpr (Lut & 0x80) r |= ( a &  b &  c);

    return r;
}

// Portable equivalent of PTX shf.r.wrap.b32.
EXL3_HD std::uint32_t funnelshift_r_wrap(
    std::uint32_t lo,
    std::uint32_t hi,
    unsigned int shift
)
{
    shift &= 31u;

    const std::uint64_t value =
        (static_cast<std::uint64_t>(hi) << 32) |
        static_cast<std::uint64_t>(lo);

    return static_cast<std::uint32_t>(value >> shift);
}

// Used by EXL3's generic fshift helper, which treats hi:lo
// as one 64-bit quantity rather than wrapping at 32 bits.
EXL3_HD std::uint32_t concat_shift_right(
    std::uint32_t lo,
    std::uint32_t hi,
    unsigned int shift
)
{
    if (shift >= 64u)
        return 0u;

    const std::uint64_t value =
        (static_cast<std::uint64_t>(hi) << 32) |
        static_cast<std::uint64_t>(lo);

    return static_cast<std::uint32_t>(value >> shift);
}

EXL3_HD std::uint32_t bfe_u32(
    std::uint32_t value,
    unsigned int offset,
    unsigned int length
)
{
    if (length == 0u || offset >= 32u)
        return 0u;

    const unsigned int available = 32u - offset;
    if (length > available)
        length = available;

    const std::uint32_t mask =
        length == 32u
            ? 0xffffffffu
            : ((std::uint32_t{1} << length) - 1u);

    return (value >> offset) & mask;
}

EXL3_HD std::uint32_t bfe_u64(
    std::uint32_t lo,
    std::uint32_t hi,
    unsigned int offset,
    unsigned int length
)
{
    if (length == 0u || offset >= 64u)
        return 0u;

    const std::uint64_t value =
        (static_cast<std::uint64_t>(hi) << 32) |
        static_cast<std::uint64_t>(lo);

    const unsigned int available = 64u - offset;
    if (length > available)
        length = available;

    const std::uint64_t mask =
        length == 64u
            ? ~std::uint64_t{0}
            : ((std::uint64_t{1} << length) - 1u);

    return static_cast<std::uint32_t>((value >> offset) & mask);
}

// EXL3 uses __dp4a(x, 0x01010101u, acc), so each unsigned
// byte of x is multiplied by one and accumulated.
EXL3_HD std::uint32_t dp4a_ones_u8(
    std::uint32_t x,
    std::uint32_t acc
)
{
    return acc
        + ((x >>  0) & 0xffu)
        + ((x >>  8) & 0xffu)
        + ((x >> 16) & 0xffu)
        + ((x >> 24) & 0xffu);
}

} // namespace exl3_rocm

#undef EXL3_HD
