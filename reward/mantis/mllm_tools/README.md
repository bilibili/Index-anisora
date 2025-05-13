## Usage

- Each `{model_name}_eval.py` can be run directly by `python {model_name}_eval.py` to roughly check the outputs on 3 examples. This is used for debugging purposes.

- Each `{model_name}_eval.py` defines a mllm model class which has a `__init__` method that takes in `model_id` for which checkpoint to load. This class also should have a `__call__` functions which takes in a list of messages in the following format:

- Please check `__init__.py` for a full list of supportted models.


## Example of adding a new model

- in `{model_name}_eval.py`:
```python

class NewModel():
    # support_multi_image: 'merge' images for False, and use custom image 'sequence' format for True
    support_multi_image = True 
    def __init__(self, model_id:str="HuggingFaceM4/idefics2-8b") -> None:
        """

        Args:
            model_path (str): model name
        """
        # load models and processors
        self.model = ...
        self.processor = ...
        
    def __call__(self, inputs: List[dict]) -> str:
        """
        Generate text from images and text prompt. (batch_size=1) one text at a time.
        Args:
            inputs (List[dict]): [
                {
                    "type": "image",
                    "content": "https://chromaica.github.io/Museum/ImagenHub_Text-Guided_IE/input/sample_34_1.jpg"
                },
                {
                    "type": "image",
                    "content": "https://chromaica.github.io/Museum/ImagenHub_Text-Guided_IE/input/sample_337180_3.jpg"
                },
                {
                    "type": "text",
                    "content": "What is difference between two images?"
                }
            ]
            Supports any form of interleaved format of image and text.
        Returns:
            str: generated text
        """
        if self.support_multi_image:
            # process images and texts ...
            generation_kwargs = {
                "max_new_tokens": 4096,
                "num_beams": 1,
                "do_sample": False,
            }
            generated_text = self.model.generate(..., generation_kwargs)
            return generated_text
        else:
            raise NotImplementedError

```

- in `__init__.py`:
```python
MLLM_LIST = [..., "NewModel"]
...
def MLLM_Models(model_name:str):
    if ...:
        ...
    elif model_name == "NewModel":
        from .new_model_eval import NewModel
        return NewModel
    else:
        raise NotImplementedError
```


## Special requirements for evaluating some models
Due to the bad compatibility of some models, we need to install some additional packages, or use a brand new environment to run the evaluation code. We list their installation requirements below:
(note, assume you are in the same directory as this README.md, which is `mantis/mllm_tools/`)

### [VILA](https://github.com/Efficient-Large-Model/VILA)
```bash
mkdir -p model_utils && cd model_utils
git clone https://github.com/Efficient-Large-Model/VILA.git

cd VILA
conda create -n vila python=3.10
conda activate vila

pip install --upgrade pip  # enable PEP 660 support
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.4.2/flash_attn-2.4.2+cu118torch2.0cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install flash_attn-2.4.2+cu118torch2.0cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install -e .
pip install -e ".[train]"

site_pkg_path=$(python -c 'import site; print(site.getsitepackages()[0])')
cp -rv ./llava/train/transformers_replace/* $site_pkg_path/transformers/
cd ../../

# then install mantis for eval, in the root directory of the repo
cd ../../
pip install -e ".[eval]"
```

After that, you need to run all other evaluation scripts in the `vila` environment.

### [Otter](https://github.com/Luodian/Otter)
```bash
mkdir -p model_utils && cd model_utils && git clone https://github.com/Luodian/Otter.git
```

Then you can test running by
```bash
python otterimage_eval.py
python ottervideo_eval.py
```

### [VideoLLaVA]
```bash
mkdir -p model_utils && cd model_utils && git clone https://github.com/PKU-YuanGroup/Video-LLaVA.git

cd Video-LLaVA
conda create -n videollava python=3.9
pip install -e .
pip install opencv-python decord pythonvideo

# then install mantis for eval, in the root directory of the repo
cd ../../
pip install -e ".[eval]"
```