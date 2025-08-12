from sat.model import ChatGLM2Model
from sat import get_args

args = get_args()

model, args = ChatGLM2Model.from_pretrained('chatglm2-6b', args)

from chat_model import ChatModel

model = ChatModel(args, model=model)
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True)
model = model.eval()
response, history = model.chat(tokenizer, "你好", history=[])
print(response)
response, history = model.chat(tokenizer, "晚上睡不着应该怎么办", history=history)
print(response)
pred = model.batch_generate(tokenizer, ["今天天气怎么样？", "你好"])
print(pred)

breakpoint()