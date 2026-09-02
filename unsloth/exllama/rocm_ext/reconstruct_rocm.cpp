// EXL3-ROCm reconstruction binding. See NOTICE.md and LICENSE.exllamav3.
#include <torch/extension.h>

#include "reconstruct_rocm.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def(
        "reconstruct",
        &reconstruct_rocm,
        "EXL3 weight reconstruction for ROCm"
    );
}
