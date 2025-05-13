from .modeling_qwen2_vl_vae import Qwen2VLVAEForConditionalGeneration
from .configuration_qwen2_vl_vae import Qwen2VLVAEConfig, Qwen2VLVisionVAEConfig
from .processing_qwen2_vl_vae import Qwen2VLVAEProcessor
from .image_processing_qwen2_vl_vae import Qwen2VLVAEImageProcessor



# Copyright 2024 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import TYPE_CHECKING
import transformers
from transformers.utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available, is_vision_available
transformers.Qwen2VLVAEImageProcessor = Qwen2VLVAEImageProcessor
transformers.Qwen2VLVAEForConditionalGeneration = Qwen2VLVAEForConditionalGeneration
transformers.Qwen2VLVAEConfig = Qwen2VLVAEConfig
transformers.Qwen2VLVAEProcessor = Qwen2VLVAEProcessor



_import_structure = {
    "configuration_qwen2_vl_vae": ["Qwen2VLVAEConfig", "Qwen2VLVisionVAEConfig"],
    "processing_qwen2_vl_vae": ["Qwen2VLVAEProcessor"],
}


try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_qwen2_vl_vae"] = [
        "Qwen2VLVisionVAEPretrainedModel",
        "Qwen2VLVAEPreTrainedModel",
        "Qwen2VLVAEForConditionalGeneration"
    ]

try:
    if not is_vision_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["image_processing_qwen2_vl_vae"] = ["Qwen2VLVAEImageProcessor"]


if TYPE_CHECKING:
    from .configuration_qwen2_vl_vae import Qwen2VLVAEConfig
    from .processing_qwen2_vl_vae import Qwen2VLVAEProcessor

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_qwen2_vl_vae import (
            Qwen2VLVAEForConditionalGeneration,
            Qwen2VLVAEPreTrainedModel,
        )

    try:
        if not is_vision_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .image_processing_qwen2_vl_vae import Qwen2VLVAEImageProcessor


else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure)
