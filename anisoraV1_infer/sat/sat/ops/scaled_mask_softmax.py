try:
    from apex.transformer.functional import FusedScaleMaskSoftmax
    from apex.transformer.enums import AttnMaskType
except ModuleNotFoundError:
    from sat.helpers import print_rank0
    print_rank0(
        "Please install apex to use FusedScaleMaskSoftmax, otherwise the inference efficiency will be greatly reduced"
    )
    FusedScaleMaskSoftmax = None
    AttnMaskType = None