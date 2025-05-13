# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Functions for downloading pre-trained DiT models
"""
import json
import os

import torch
from torchvision.datasets.utils import download_url

pretrained_models = {"DiT-XL-2-512x512.pt", "DiT-XL-2-256x256.pt"}


def find_model(model_name):
    """
    Finds a pre-trained DiT model, downloading it if necessary. Alternatively, loads a model from a local path.
    """
    if model_name in pretrained_models:  # Find/download our pre-trained DiT checkpoints
        return download_model(model_name)
    else:  # Load a custom DiT checkpoint:
        if not os.path.isfile(model_name):
            # if the model_name is a directory, then we assume we should load it in the Hugging Face manner
            # i.e. the model weights are sharded into multiple files and there is an index.json file
            # walk through the files in the directory and find the index.json file
            index_file = [os.path.join(model_name, f) for f in os.listdir(model_name) if "index.json" in f]
            assert len(index_file) == 1, f"Could not find index.json in {model_name}"

            # process index json
            with open(index_file[0], "r") as f:
                index_data = json.load(f)

            bin_to_weight_mapping = dict()
            for k, v in index_data["weight_map"].items():
                if v in bin_to_weight_mapping:
                    bin_to_weight_mapping[v].append(k)
                else:
                    bin_to_weight_mapping[v] = [k]

            # make state dict
            state_dict = dict()
            for bin_name, weight_list in bin_to_weight_mapping.items():
                bin_path = os.path.join(model_name, bin_name)
                bin_state_dict = torch.load(bin_path, map_location=lambda storage, loc: storage)
                for weight in weight_list:
                    state_dict[weight] = bin_state_dict[weight]
            return state_dict
        else:
            # if it is a file, we just load it directly in the typical PyTorch manner
            assert os.path.exists(model_name), f"Could not find DiT checkpoint at {model_name}"
            checkpoint = torch.load(model_name, map_location=lambda storage, loc: storage)
            if "ema" in checkpoint:  # supports checkpoints from train.py
                checkpoint = checkpoint["ema"]
            return checkpoint


def download_model(model_name):
    """
    Downloads a pre-trained DiT model from the web.
    """
    assert model_name in pretrained_models
    local_path = f"pretrained_models/{model_name}"
    if not os.path.isfile(local_path):
        os.makedirs("pretrained_models", exist_ok=True)
        web_path = f"https://dl.fbaipublicfiles.com/DiT/models/{model_name}"
        download_url(web_path, "pretrained_models")
    model = torch.load(local_path, map_location=lambda storage, loc: storage)
    return model


if __name__ == "__main__":
    # Download all DiT checkpoints
    for model in pretrained_models:
        download_model(model)
    print("Done.")
