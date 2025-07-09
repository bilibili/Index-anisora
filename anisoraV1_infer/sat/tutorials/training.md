# The Training tutorial
## The Training API
We provide a simple but powerful training API `training_main()`, which is not limited to our Transformer models, but also applicable to any `torch.nn.Module`. 
```python
from sat import get_args, training_main
from sat.model import AutoModel, BaseModel
args = get_args()
# to pretrain from scratch, give a class obj
model = BaseModel
# to finetuned from a given model, give a torch.nn.Module
model = AutoModel.from_pretrained('bert-base-uncased', args)

training_main(args, 
    model_cls=model,
    forward_step_function=forward_step,
    create_dataset_function=dataset_func,
    handle_metrics_function=None,
    init_function=None
)
```
The above is an (incomplete) example for a standard training program using `sat`. The `training_main` accepts 5 parameters:
* (Required) `model_cls`: a type object inheriting `torch.nn.Module`, or a `torch.nn.Module` object which we train from. 
* (Required) `forward_step_function`: a customized function with input `data_iterator, model, args, timers`, returns `loss, {'metric0': m0, ...}`.
* (Required) `create_dataset_function`: Return a `torch.utils.data.Dataset` for loading. Our library will automatically distribute the data into multiple workers, and give the dataiterator to `forward_step_function`.
* (Optional) `handle_metrics_function`: handle special metrics during evaluation. 
* (Optional) `init_function`: a hook to change model exactly before training, useful in continuing training.

See [Finetune BERT example](../examples/bert/finetune_bert_boolq.py)for a complete example.


## Dataset
The `create_dataset_function` has inputs of `path, args` where the path could be either `args.train_data` or `args.valid_data` or `arg.test_data`. 

The function returns a `torch.utils.data.Dataset`. 

**What does sat do about the dataset?** It splits the dataset into different works and randomly shuffles it (without saving the large id mapping list). It also duplicates the dataset to make it enough to train `args.train_iters` iterations. 

If you don't have a separated validation set (usually in pretraining), you can pass `--split 95,5,5` to split the training set to train/valid/test sets according to this ratio.

### Huggingface datasets
We provide a quick entry to load huggingface datasets `load_hf_dataset(path, process_fn, columns, cache_dir, offline=False, rebuild=False, transformer_name=somemodel)`. The path should be like `hf://huggingface_path/to/dataset`, and `transformer_name` is a identifier. 

This function will process each sample with `process_fn` and save it to the disk for cache (use the `transformer_name` as a identifier), pass `rebulid=True` when changing `process_fn`. 

To use this quick entry is **not necessary**, it is valid as long as you return a `torch.utils.data.Dataset`.

## Forward Step
The `forward_step` is the customized function to use the model to compute loss. You don't need to care about the back propagation. 
A forward_step usually looks like
```python

def forward_step(data_iterator, model, args, timers):
    """Forward step."""
    # Get the data.
    raw_inputs = next(data_iterator)

    # processing, e.g. transform input to fp16 according to args.fp16
    tokens, labels, attention_mask, position_ids, token_type_ids, loss_mask = raw_inputs
    if args.fp16:
        attention_mask = attention_mask.half()
    
    # forward model
    logits, *others = model(input_ids=tokens, position_ids=position_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
    pred = logits.contiguous().float().squeeze(-1)[..., 0]
    # compute loss
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        pred,
        labels.float()
    )
    # other metrics
    acc = ((pred > 0.).long() == labels).sum() / labels.numel()
    # the second returned value is a dict containing other metrics, will be averaged and logged during training.
    return loss, {'acc': acc}
```
This is flexible enough to customize the your training.  

## Hyperparameters
There are many hyperparameters to control the training, see `add_training_args(parser)` in the [arguments.py](/sat/arguments.py).

Very common ones: 
`--train-iters`, `--batch-size`, `--lr`, `--seed`, `--zero-stage`, `--checkpoint-activations`, `--fp16`, `--gradient-accumulation-steps`, `--warmup`...

## Save
The `--save-interval` means that every `args.save_interval` iterations to save once during training. 

Note we never save the optimizer states due to the high disk consumption and slow loading. We find a simple warmup always give the same performance when continuing training.

The `--save` controls the path (under which folder) to store the saved checkpoints.

## Load 
There two ways to load an existing model: 

The first is `ModelClass.from_pretrained(name, args, url=None, home_path=None)`. It will create an object of `ModelClass` and load the checkpoint in `home_path/name`.