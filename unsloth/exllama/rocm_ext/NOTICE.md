# EXL3 ROCm reconstruction source provenance

These six native inputs were copied from the EXL3-ROCm project's
`experiments/single-layer/rocm_ext/` implementation. Before the attribution
comments were added, all six were byte-identical to those research files.
The reconstruction algorithm has not been changed during packaging.

Reference inspected: ExLlamaV3 commit
`0c49587a7c235e6303a6bbedc8b665272ad3a2ea`, including its `LICENSE` and
`exllamav3/exllamav3_ext/` sources. This identifies the reference used for
this provenance review, not a claim about the original port's starting commit.
The upstream copyright and complete MIT permission notice are retained in
`LICENSE.exllamav3` alongside this file.

- `portable_codebook.h` adapts the constants and integer transformations in
  upstream `quant/codebook.cuh` (`decode_3inst`). The EXL3-ROCm implementation
  was introduced in commit `14c0551f46f1315875c04e6375ecb8ffab2f8a21`.
- `portable_decode_device.cuh` adapts the scalar half decoder from
  `quant/codebook.cuh` and the trellis-window formula from `quant/exl3_dq.cuh`,
  replacing CUDA/PTX operations with HIP and portable helpers.
- `reconstruct_rocm.cu` implements the upstream `quant/reconstruct.cu` packed
  format and reconstruction layout using direct scalar stores instead of the
  upstream warp-shuffle implementation. It adds project-specific HIP dispatch
  and validation. This kernel and the device decoder were introduced in
  EXL3-ROCm commit `c495e95be20f4a43f19dddcd960b4bc47c6addf7`.
- `portable_bitops.h` contains project-written C++ equivalents of the bit
  operations used by the upstream decoder, rather than copied PTX assembly.
  Its introduction is recorded in EXL3-ROCm commit
  `0a6d10f871556bed7583c899355c14cd16244ca3`.
- `reconstruct_rocm.cpp` and `reconstruct_rocm.h` are the project's narrow
  PyTorch binding and function declaration, introduced in EXL3-ROCm commit
  `8b45aef`.

The project-specific glue and portable implementations above are distinguished
from the adapted upstream algorithm; this record does not assert independent
ownership of the complete reconstruction implementation or relicense the
upstream portions. Both this record and the upstream license ship as package
data. Host-oracle sources and generated HIP/build artifacts are not packaged.
