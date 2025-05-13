import inspect
from abc import abstractmethod
from dataclasses import dataclass

import torch
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.utils import BaseOutput


class VideoSysPipeline(DiffusionPipeline):
    def __init__(self):
        super().__init__()

    @staticmethod
    def set_eval_and_device(device: torch.device, *modules):
        modules = list(modules)
        for i in range(len(modules)):
            modules[i] = modules[i].eval()
            modules[i] = modules[i].to(device)

    @abstractmethod
    def generate(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        """
        In diffusers, it is a convention to call the pipeline object.
        But in VideoSys, we will use the generate method for better prompt.
        This is a wrapper for the generate method to support the diffusers usage.
        """
        return self.generate(*args, **kwargs)

    @classmethod
    def _get_signature_keys(cls, obj):
        parameters = inspect.signature(obj.__init__).parameters
        required_parameters = {k: v for k, v in parameters.items() if v.default == inspect._empty}
        optional_parameters = set({k for k, v in parameters.items() if v.default != inspect._empty})
        expected_modules = set(required_parameters.keys()) - {"self"}
        # modify: remove the config module from the expected modules
        expected_modules = expected_modules - {"config"}

        optional_names = list(optional_parameters)
        for name in optional_names:
            if name in cls._optional_components:
                expected_modules.add(name)
                optional_parameters.remove(name)

        return expected_modules, optional_parameters


@dataclass
class VideoSysPipelineOutput(BaseOutput):
    video: torch.Tensor
