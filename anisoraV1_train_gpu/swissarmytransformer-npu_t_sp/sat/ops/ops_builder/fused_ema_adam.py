from .builder import CUDAOpBuilder

import sys


class FusedEmaAdamBuilder(CUDAOpBuilder):
    BUILD_VAR = "SAT_BUILD_FUSED_EMA_ADAM"
    NAME = "fused_ema_adam"

    def __init__(self):
        super().__init__(name=self.NAME)

    def absolute_name(self):
        return f'sat.ops.{self.NAME}_op'

    def sources(self):
        return ['csrc/adam/fused_ema_adam_frontend.cpp', 'csrc/adam/multi_tensor_ema_adam.cu']

    def include_paths(self):
        return ['csrc/includes', 'csrc/adam']

    def cxx_args(self):
        args = super().cxx_args()
        return args + self.version_dependent_macros()

    def nvcc_args(self):
        nvcc_flags = ['-O3'] + self.version_dependent_macros()
        if not self.is_rocm_pytorch():
            nvcc_flags.extend(
                ['-allow-unsupported-compiler' if sys.platform == "win32" else '', '-lineinfo', '--use_fast_math'] +
                self.compute_capability_args())
        return nvcc_flags
