# 2021.10.29 v0.1
1. change `mixins` from `ModuleList` to `ModuleDict`
2. return tokens and mems in `fill_sequence`, and mems becomes a tensor.
3. `CachedAutoRegressiveMixin`
## How to migrate old SAT ckpt to new version?
Example:
```python
import torch
old = torch.load('xxxxx/mp_rank_00_model_states.pt.old', map_location='cpu')

# replace names, mixins index to keys
oldm = old['module']
for k in list(oldm.keys()):
    if k.startswith('mixins.0'):
        new_k = k.replace('mixins.0', 'mixins.extra_position_embedding')
    elif k.startswith('mixins.1'):
        new_k = k.replace('mixins.1', 'mixins.attention_plus')
    else:
        continue
    oldm[new_k] = oldm[k]
    del oldm[k]
# save to destination    
torch.save(old, 'xxxxx/mp_rank_00_model_states.pt')

```
for the older framework, you also need:
```python
old['module']['transformer.word_embeddings.weight'] = old['module']['word_embeddings.weight']
del old['module']['word_embeddings.weight']
```
# 2021.11.5 v0.1.2
1. Add generation.autoregressive_sampling.evalute_perplexity
2. fix Runtime Error in skipping Nan Loss

# 2021.12.13 v0.1.4
1. Add non_conflict attention_fn
2. Add Prefix-Tuning
3. Now, you can use `kw_args['output_this_layer']` (any hooks in the transformer layers) to return values to final outputs and `kw_args['output_cross_layer']` to pass values to `kw_args` in the next layer.

Examples:
```
def attention_fn(...some_args):
    ...
    kw_args['output_this_layer']['mem_kv'] = cache_kv
    ...
```
This will let the key `'mem_kv'` appear in the `outputs_per_layers[i]` of `logits, *outputs_per_layers = model(...)`. 

```
def attention_fn(...some_args, **kw_args):
    ...
    kw_args['output_cross_layer']['last_attention_map'] = attention_map
    ...
```
This will let the key `'last_attention_map'` appear in the next layer's `kw_args` (all hooks). 

# 2021.12.13 v0.1.7
1. Ensure enough training data, no longer always 200 times
2. You can use `kw_args['cross_layer_output']['new_key']=xxx` to pass other results to each layer in `position/word_embedding_forward`.
3. Add `--train-data-weights`.

# 2022.1.13 v0.1.9
1. Add Vit
2. Fix evaluation all_reduce bug

# 2022.6.3 v2.0
1. split all the default hooks out
2. change the order, model hooks will not override all the things. They now are the same as mixin hooks added in the **front** of all the mixins.

# 2022.6.6 v2.0
1. `from_pretrained` now auto downloads models. There are two kinds of usages: `SomeModel.from_pretrained(args, name)` will load the weights of `name` model to a `SomeModel` with the same model arch hyper-params with `name`; `AutoModel.from_pretrained(args, name)` will return an official model (`model_class` Class) with the pretrained weights.
2. ENV `SAT_HOME` is where we put the models in. Set it in your shell file.
3. don't necessarily need `deepspeed_config`, or pass model arch hyper-params for `from_pretrained`. Use `zero-stage 0/1/2`.  

# 2022.6.27
1. Fix *flat_output bug.
2. fix defualt mpu init_method bug.

# 2023.4.9
Large update v.0.3.0
1. delete `--sandwich-ln`
2. `from_pretrained(args, name) => from_pretrained(name, args=None)`
3. MODEL_URLS fix typo
4. enable model-only mode

# 2023.4.11
v.0.3.1
refactor SwissArmyTransformer as sat (package name SwissArmyTransformer)

# 2023.4.21
v 0.3.2
fix model-only "create then inference" bug
support deepspeed 0.8.x & 0.9.x
model register first try

# 2023.4.23
v 0.3.3
change the fp16 & to cuda order in `get_model`.

# 2023.5.15
v 0.3.4
1. add example for nested transformer models
2. move all print to logging, set `SAT_LOGLEVEL` to control

# 2023.5.16
v. 0.3.5
1. add repetition penalty
2. add quantization

# 2023.5.17
v. 0.3.6
support no deepspeed model-only
test cpu inference
test windows

# 2023.6.1
v. 0.3.7
update vit
add qlora/lora2

# 2023.7.3
v. 0.4.0
1. add xfomers memory efficient attention.
2. pytorch 2.0 auto fast attention, attention_fn dispatch via version.
3. add llama and chatglm2.
4. add split model for model-parallel in inference mode.
5. add r2 download

# 2023.7.13
v. 0.4.1
1. better model parallel support (training mode split)
2. better default zero 1/2 config
3. test bf16 training
4. change qkv order of chatglm1
5. only use pytorch 2.0 attention when full / causal.

# 2023.9.10
v. 0.4.6
1. add droppath and checkpoint last layer skip
2. support multiple webdataset weighting
3. fix lora merging
4. add different lr in different parts, add a 'lr' attr for parameters in the `disable_untrainable_params`.

# 2024.1.11
v. 0.4.10
1. fix model parallel init possible bug by additional broadcast
2. add nsys profiling
3. add gated mlp option
4. support batch_from_same_dataset for multi-webds
5. fix cmp kernel quant no bias bug

# 2024.1.18
v. 0.4.11
1. fix the tarfile buffer_size bug in 0.4.9 and 0.4.10.
2. fix potential problem to pass a mixed-device model to training_main
3. fix emaadam no use error introduced in 0.4.9 and 0.4.10.