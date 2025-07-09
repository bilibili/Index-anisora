# SAT 模型使用
sat 库支持对于预训练模型的自动下载和使用，全部的预训练模型名称在[urls.py](/sat/resources/urls.py)中，所有的预训练模型使用相同的基本骨架，可以快速地在其基础上进行改动（例如增加Lora等）。设置环境变量`SAT_HOME=/path/to/save`来决定模型存储位置，默认为`~/.sat_models`。

## 预训练模型加载
下面这段代码是加载`bert-base`模型的示例。
```python
from sat.model import AutoModel
model, args = AutoModel.from_pretrained('roberta-base')
x = torch.tensor([[1,2,3]], device='cuda') # fake input for test
a = model(input_ids=x, position_ids=x, attention_mask=None)
```
`from_pretrained`函数的第一个参数接受模型文件夹路径、zip包URL、可下载模型名称等。

`from_pretrained`函数返回模型和预训练模型参数`args`，这里一个`args`是`argparse.Namespace`的实例。例如在上述实例中输出的args为
```python
Namespace(
    model_class='RobertaModel', model_parallel_size=1, num_attention_heads=12, num_layers=12, ... # others, not displayed
)
```

## 模型参数
sat中，模型参数大部分都是以`argparse.Namespace`的格式出现的，某类模型的参数可以通过`list_avail_args`查看。
```

```
在磁盘上以Json格式存储。具体地，如果是路径，则它指向一个文件夹，模型参数存储在该文件夹下的`model_config.json`中。




