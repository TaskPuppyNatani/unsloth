// ROCm implementation of the ExLlamaV3 reconstruction layout.
// Upstream portions: Copyright (c) 2025 Turboderp; see LICENSE.exllamav3 and NOTICE.md.
#include <array>
#include <cstdint>

#include <ATen/Tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Exception.h>

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "portable_decode_device.cuh"
#include "reconstruct_rocm.h"


template <int Bits, int Codebook>
__global__
void reconstruct_kernel_correctness(
    __half* __restrict__ unpacked,
    const std::uint16_t* __restrict__ packed,
    int packed_blocks_n,
    int output_cols
)
{
    constexpr int packed_size =
        256 * Bits / 16;

    const int t =
        static_cast<int>(threadIdx.x);

    if (t >= 256)
        return;

    const int logical_group =
        t / 32;

    const int lane =
        t % 32;

    const int row_block =
        static_cast<int>(blockIdx.y);

    const int col_block =
        static_cast<int>(blockIdx.x);

    const int packed_col =
        col_block * 8 +
        logical_group;

    const std::uint16_t* packed_u16 =
        packed +
        (
            row_block * packed_blocks_n +
            packed_col
        ) *
        packed_size;

    const std::uint32_t* packed_u32 =
        reinterpret_cast<
            const std::uint32_t*
        >(
            packed_u16
        );

    const int r0 =
        (lane % 4) * 2;

    const int r1 =
        r0 + 1;

    const int r2 =
        r0 + 8;

    const int r3 =
        r0 + 9;

    const int c0 =
        lane / 4;

    const int c1 =
        c0 + 8;

    const int base_output_col =
        col_block * 128 +
        logical_group * 16;

    const int base_offset =
        lane * 8;

    const __half v0 =
        exl3_rocm::dq_device<
            Bits,
            Codebook
        >(
            packed_u32,
            base_offset + 0
        );

    const __half v1 =
        exl3_rocm::dq_device<
            Bits,
            Codebook
        >(
            packed_u32,
            base_offset + 1
        );

    const __half v2 =
        exl3_rocm::dq_device<
            Bits,
            Codebook
        >(
            packed_u32,
            base_offset + 2
        );

    const __half v3 =
        exl3_rocm::dq_device<
            Bits,
            Codebook
        >(
            packed_u32,
            base_offset + 3
        );

    const __half v4 =
        exl3_rocm::dq_device<
            Bits,
            Codebook
        >(
            packed_u32,
            base_offset + 4
        );

    const __half v5 =
        exl3_rocm::dq_device<
            Bits,
            Codebook
        >(
            packed_u32,
            base_offset + 5
        );

    const __half v6 =
        exl3_rocm::dq_device<
            Bits,
            Codebook
        >(
            packed_u32,
            base_offset + 6
        );

    const __half v7 =
        exl3_rocm::dq_device<
            Bits,
            Codebook
        >(
            packed_u32,
            base_offset + 7
        );

    const int output_row_base =
        row_block * 16;

    unpacked[
        (output_row_base + r0) *
        output_cols +
        base_output_col +
        c0
    ] = v0;

    unpacked[
        (output_row_base + r1) *
        output_cols +
        base_output_col +
        c0
    ] = v1;

    unpacked[
        (output_row_base + r2) *
        output_cols +
        base_output_col +
        c0
    ] = v2;

    unpacked[
        (output_row_base + r3) *
        output_cols +
        base_output_col +
        c0
    ] = v3;

    unpacked[
        (output_row_base + r0) *
        output_cols +
        base_output_col +
        c1
    ] = v4;

    unpacked[
        (output_row_base + r1) *
        output_cols +
        base_output_col +
        c1
    ] = v5;

    unpacked[
        (output_row_base + r2) *
        output_cols +
        base_output_col +
        c1
    ] = v6;

    unpacked[
        (output_row_base + r3) *
        output_cols +
        base_output_col +
        c1
    ] = v7;
}


template <int Codebook>
void launch_reconstruct(
    int bits,
    dim3 grid,
    dim3 block,
    decltype(
        at::cuda::getCurrentCUDAStream()
    ) stream,
    __half* unpacked,
    const std::uint16_t* packed,
    int packed_blocks_n,
    int output_cols
)
{
#define EXL3_LAUNCH(BITS)                                  \
    reconstruct_kernel_correctness<                        \
        BITS,                                              \
        Codebook                                           \
    ><<<                                                   \
        grid,                                              \
        block,                                             \
        0,                                                 \
        stream.stream()                                    \
    >>>(                                                   \
        unpacked,                                          \
        packed,                                            \
        packed_blocks_n,                                   \
        output_cols                                        \
    )

    switch (bits)
    {
        case 1: EXL3_LAUNCH(1); break;
        case 2: EXL3_LAUNCH(2); break;
        case 3: EXL3_LAUNCH(3); break;
        case 4: EXL3_LAUNCH(4); break;
        case 5: EXL3_LAUNCH(5); break;
        case 6: EXL3_LAUNCH(6); break;
        case 7: EXL3_LAUNCH(7); break;
        case 8: EXL3_LAUNCH(8); break;

        default:
            TORCH_CHECK(
                false,
                "reconstruct_rocm: K must be between 1 and 8"
            );
    }

#undef EXL3_LAUNCH
}


void reconstruct_rocm(
    at::Tensor unpacked,
    at::Tensor packed,
    int K,
    bool mcg,
    bool mul1
)
{
    TORCH_CHECK(
        unpacked.is_cuda(),
        "reconstruct_rocm: unpacked must be a GPU tensor"
    );

    TORCH_CHECK(
        packed.is_cuda(),
        "reconstruct_rocm: packed must be a GPU tensor"
    );

    TORCH_CHECK(
        unpacked.device() == packed.device(),
        "reconstruct_rocm: tensors must be on the same device"
    );

    TORCH_CHECK(
        unpacked.is_contiguous(),
        "reconstruct_rocm: unpacked must be contiguous"
    );

    TORCH_CHECK(
        packed.is_contiguous(),
        "reconstruct_rocm: packed must be contiguous"
    );

    TORCH_CHECK(
        unpacked.dim() == 2,
        "reconstruct_rocm: unpacked must be 2D"
    );

    TORCH_CHECK(
        packed.dim() == 3,
        "reconstruct_rocm: packed must be 3D"
    );

    TORCH_CHECK(
        unpacked.scalar_type() == at::kHalf,
        "reconstruct_rocm: unpacked must be FP16"
    );

    TORCH_CHECK(
        K >= 1 && K <= 8,
        "reconstruct_rocm: K must be between 1 and 8"
    );

    TORCH_CHECK(
        !(mcg && mul1),
        "reconstruct_rocm: mcg and mul1 cannot both be enabled"
    );

    TORCH_CHECK(
        packed.size(2) == 256 * K / 16,
        "reconstruct_rocm: invalid packed EXL3 block size"
    );

    TORCH_CHECK(
        unpacked.size(0) == packed.size(0) * 16,
        "reconstruct_rocm: invalid output row count"
    );

    TORCH_CHECK(
        unpacked.size(1) == packed.size(1) * 16,
        "reconstruct_rocm: invalid output column count"
    );

    TORCH_CHECK(
        unpacked.size(1) % 128 == 0,
        "reconstruct_rocm: output columns must be divisible by 128"
    );

    if (unpacked.numel() == 0)
        return;

    const at::cuda::OptionalCUDAGuard device_guard(
        unpacked.device()
    );

    auto stream =
        at::cuda::getCurrentCUDAStream();

    const dim3 block(256);

    const dim3 grid(
        static_cast<unsigned int>(
            unpacked.size(1) / 128
        ),
        static_cast<unsigned int>(
            packed.size(0)
        )
    );

    auto* unpacked_ptr =
        reinterpret_cast<__half*>(
            unpacked.data_ptr()
        );

    const auto* packed_ptr =
        reinterpret_cast<
            const std::uint16_t*
        >(
            packed.data_ptr()
        );

    if (mcg)
    {
        launch_reconstruct<1>(
            K,
            grid,
            block,
            stream,
            unpacked_ptr,
            packed_ptr,
            static_cast<int>(
                packed.size(1)
            ),
            static_cast<int>(
                unpacked.size(1)
            )
        );
    }
    else if (mul1)
    {
        launch_reconstruct<2>(
            K,
            grid,
            block,
            stream,
            unpacked_ptr,
            packed_ptr,
            static_cast<int>(
                packed.size(1)
            ),
            static_cast<int>(
                unpacked.size(1)
            )
        );
    }
    else
    {
        launch_reconstruct<0>(
            K,
            grid,
            block,
            stream,
            unpacked_ptr,
            packed_ptr,
            static_cast<int>(
                packed.size(1)
            ),
            static_cast<int>(
                unpacked.size(1)
            )
        );
    }

    const hipError_t error =
        hipGetLastError();

    TORCH_CHECK(
        error == hipSuccess,
        "reconstruct_rocm: HIP kernel launch failed: ",
        hipGetErrorString(error)
    );
}
