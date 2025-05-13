# Introduction
`sat`(`SwissArmyTransformer`) is a flexible and powerful library to develop your own Transformer variants.

`sat` is named after "swiss army knife", meaning that all the models (e.g. BERT, GPT, T5, GLM, CogView, ViT...) **share the same backone code** and cater for versatile usages with some extra light-weight mixins. 

`sat` is powered by `deepspeed-ZeRO` and model parallelism, aiming to provide the best practice for pretraining and finetuning large models (100M\~20B parameters). 

## Install
```
    pip install SwissArmyTransformer
```
# Features

* **Add model-agnostic components**, e.g. prefix-tuning, in just *ONE* line! 

    - [Prefix-tuning](https://arxiv.org/pdf/2101.00190) (or [P-tuning](https://arxiv.org/abs/2103.10385)) improves finetuning via adding trainable parameters in each attention layer. To apply it to a [GLM](https://arxiv.org/pdf/2103.10360.pdf) classification (or any other) model is easy with our library.

    ```python
        class ClassificationModel(GLMModel): # can also be BertModel, RobertaModel, etc. 
            def __init__(self, args, transformer=None, **kwargs):
                super().__init__(args, transformer=transformer, **kwargs)
                self.add_mixin('classification_head', MLPHeadMixin(args.hidden_size, 2048, 1))
                # Arm an arbitrary model with Prefix-tuning with this line!
                self.add_mixin('prefix-tuning', PrefixTuningMixin(args.num_layers, args.hidden_size // args.num_attention_heads, args.num_attention_heads, args.prefix_len))
    ```

    - GPT and other auto-regressive models act differently during training and inference. During inference, text is generated token-by-token and we need to cache previous states for efficiency. With our lib, you only need to consider the behavior during training (teacher-forcing) and transform it to a cached auto-regressive model via adding a mixin:

    ```python
        model, args = AutoModel.from_pretrained('glm-10b-chinese', args)
        model.add_mixin('auto-regressive', CachedAutoregressiveMixin())
        # Generate a sequence with beam search
        from sat.generation.autoregressive_sampling import filling_sequence
        from sat.generation.sampling_strategies import BeamSearchStrategy
        output, *mems = filling_sequence(model, input_seq,
                        batch_size=args.batch_size,
                        strategy=BeamSearchStrategy(args.batch_size))
    ```     


* **Build your Transformer-based model with minimal codes**. We mentioned [GLM](https://arxiv.org/pdf/2103.10360.pdf), which only differs from standard transformer (called BaseModel) on position embedding (and training losses). We only need to focus on the related part when coding.

    <details><summary>Extend the whole definition: </summary><p>

    ```python
    class BlockPositionEmbeddingMixin(BaseMixin):
        # Here define parameters for the mixin
        def __init__(self, max_sequence_length, hidden_size, init_method_std=0.02):
            super(BlockPositionEmbeddingMixin, self).__init__()
            self.max_sequence_length = max_sequence_length
            self.hidden_size = hidden_size
            self.block_position_embeddings = torch.nn.Embedding(max_sequence_length, hidden_size)
            torch.nn.init.normal_(self.block_position_embeddings.weight, mean=0.0, std=init_method_std)
        
        # Here define the method for the mixin
        def position_embedding_forward(self, position_ids, **kwargs):
            position_ids, block_position_ids = position_ids[:, 0], position_ids[:, 1]
            position_embeddings = self.transformer.position_embeddings(position_ids)
            block_position_embeddings = self.block_position_embeddings(block_position_ids)
            return position_embeddings + block_position_embeddings

    class GLMModel(BaseModel):
        def __init__(self, args, transformer=None):
            super().__init__(args, transformer=transformer)
            self.add_mixin('block_position_embedding', 
                BlockPositionEmbeddingMixin(args.max_sequence_length, args.hidden_size)
            ) # Add the mixin for GLM
    ```

*  **Comprehensive supports for training**. `sat` aims to provide the best practice for pretraining and finetuning, where you only need to finish `forward_step` and `create_dataset_function` but with hyperparameters to alter useful training configurations.
    - Extend the training to multiple GPUs or nodes by specifying `--num_nodes`, `--num_gpus` and a simple `hostfile`. 
    - DeepSpeed and Model parallelism.
    - Better integration of ZeRO-2 and activation checkpointing.
    - Automatic extending and shuffling training data and `memmap`. 
    - Successfully support the training of [CogView2](http://github.com/THUDM/CogView2) and [CogVideo](https://github.com/THUDM/cogvideo).
    - The only open-source codebase supporting finetuning [T5-10B](https://arxiv.org/abs/1910.10683) on GPUs currently.

</p></details>


# Quick Tour

The most typical python file to use `Bert` in sat (for inference) is as follows:
```python
# @File: inference_bert.py
from sat import get_args, get_tokenizer, AutoModel
# Parse args, initialize the environment. This is necessary.
args = get_args() 
# Automatically download and load model. Will also dump model-related hyperparameters to args.
model, args = AutoModel.from_pretrained('bert-base-uncased', args) 
# Get the BertTokenizer according to args.tokenizer_type (automatically set).
tokenizer = get_tokenizer(args) 
# Here to use bert as you want!
# ...
```
Then we can run the code via
```bash
    SAT_HOME=/path/to/download python inference_bert.py --mode inference
```
All officially supported model names are in [urls.py](sat/resources/urls.py).

To finetune or pretrain a transformer is also extremely easy!
```python
# @File: finetune_bert.py
from sat import get_args, get_tokenizer, AutoModel
from sat.model.mixins import MLPHeadMixin

def create_dataset_function(path, args):
    # Here to load the dataset
    # ...
    assert isinstance(dataset, torch.utils.data.Dataset)
    return dataset

def forward_step(data_iterator, model, args, timers):
    inputs = next(data_iterator) # from the dataset of create_dataset_function.
    loss, *others = model(inputs)
    return loss
    
# Parse args, initialize the environment. This is necessary.
args = get_args() 
model, args = AutoModel.from_pretrained('bert-base-uncased', args) 
tokenizer = get_tokenizer(args) 
# Here to use bert as you want!
model.del_mixin('bert-final')
model.add_mixin('classification_head', MLPHeadMixin(args.hidden_size, 2048, 1))
# ONE LINE to train! 
# args already includes hyperparams such as lr, train-iters, zero-stage ...
training_main(args, 
    model_cls=model, 
    forward_step_function=forward_step, # user define
    create_dataset_function=create_dataset_function # user define
)
```
Then we can run the code via
```shell
deepspeed --include localhost:0,1 finetune_bert.py \
    --experiment-name ftbert \
    --mode finetune --train-iters 1000 --save /path/to/save \
    --train-data /path/to/train --valid-data /path/to/valid \
    --lr 0.00002 --batch-size 8 --zero-stage 1 --fp16
```
Here we use data-parallel on GPUs 0,1. We can also launch the training on many inter-connected machines via `--hostfile /path/to/hostfile`. See the tutorial for more details.

To write your own model, you only need to consider the difference between the standard Transformer. For example, if you have a idea to improve the attention operation:
```python
from sat.model import BaseMixin
class MyAttention(BaseMixin):
    def __init__(self, hidden_size):
        super(MyAttention, self).__init__()
        # MyAttention may needs some new params, e.g. a learnable alpha.
        self.learnable_alpha = torch.nn.Parameter(torch.ones(hidden_size))
    
    # This is a hook function, the name `attention_fn` is special.
    def attention_fn(q, k, v, mask, dropout=None, **kwargs):
        # Code for my attention.
        # ...
        return attention_results
```
Here `attention_fn` is a hook function, replacing the default action by the new function. All available hooks are in [transformer_defaults.py](/sat/transformer_defaults.py). 
Now we can use `add_mixin` to apply our change to all the transformers, such as BERT, Vit and CogView. See the tutorial for more details. 

## Tutorials 
* [How to use pretrained models collected in sat?](tutorials/model_usage)
* [Why and how to train models in sat?](tutorials/training)

# Citation
Currently we don't have a paper, so you don't need to formally cite us!~ 

If this project helps your research or engineering, use `\footnote{https://github.com/THUDM/SwissArmyTransformer}` to mention us and recommend `SwissArmyTransformer` to others.

The tutorial for contributing sat is on the way!

The project is based on (a user of) DeepSpeed, Megatron-LM and Huggingface transformers. Thanks for their awesome work.
