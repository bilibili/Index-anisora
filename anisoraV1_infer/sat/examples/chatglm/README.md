# ChatGLM

This folder provides inference and fineuning for [ChatGLM-6B](https://github.com/THUDM/ChatGLM-6B) model in SAT.

If you want to inference or chat with ChatGLM-6B:

```
python chat_sat.py
```

If you want to inference or chat with ChatGLM-6B with huggingface generation instead of sat (legacy):
```
python inference_chatglm.py --mode inference --fp16 --device 0 --skip-init
python chat.py --mode inference --fp16 --skip-init
```

Models will be downloaded and cached automatically into `~/.sat_models`.

For finetuning ChatGLM-6B, we adapt the [official finetune code](https://github.com/THUDM/ChatGLM-6B/tree/main/ptuning) to SAT, and provide two parameter-efficient tuning methods, i.e., ptuning and lora.

For ptuning:

```
bash scripts/finetune_adgen_ptuning.sh
python inference_adgen.py --mode inference --skip-init --fp16 --ckpt_path checkpoints/finetune-chatglm-6b-adgen-05-02-10-32/
```

Examples of ptuning finetuned model:

```
输入提示语：类型#上衣*材质#牛仔布*颜色#白色*风格#简约*图案#刺绣*衣样式#外套*衣款式#破洞
['类型#上衣*材质#牛仔布*颜色#白色*风格#简约*图案#刺绣*衣样式#外套*衣款式#破洞 牛仔外套以简约的白色为主色调,用
破洞和刺绣图案为外套增添时尚气息,打破单调感。破洞设计让外套更加有层次感,搭配起来更加有范。']
输入提示语：你好
['你好 小<UNK>,在<UNK>的<UNK>和<UNK>之间,有着一种神秘而美丽的关系。在<UNK>的<UNK>和<UNK>之间,有着一种神秘而美
丽的关系。在<UNK>的<UNK>和<UNK>之间,有着一种神秘而美丽的关系。']
输入提示语：北京市是什么
['北京市是什么 北京市位于中国的首都北京,有着丰富的历史文化底蕴和独特的城市风貌,拥有众多的历史古迹和文化遗产。
其中,景山、天坛等,更是有着独特的历史背景和文化底蕴,吸引着众多游客前来打卡拍照。']
输入提示语：类型#裙*材质#针织*颜色#纯色*风格#复古*风格#文艺*风格#简约*图案#格子*图案#纯色*图案#复古*裙型#背带
裙*裙长#连衣裙*裙领型#半高领
['类型#裙*材质#针织*颜色#纯色*风格#复古*风格#文艺*风格#简约*图案#格子*图案#纯色*图案#复古*裙型#背带 复古格子元
素的加入,为整个裙子带来独特的魅力,纯色的面料加上背带的设计,既舒适又文艺,尽显复古文艺范。经典的纯色与复古格子的
搭配,经典与时尚的完美结合,简约而不失气质。']
输入提示语：['裙*裙长#连衣裙*裙领型#半高领 一款经典的连衣裙,无论是什么场合,都适用。半高领的设计,可以凸显颈部线
条,彰显女性的优雅与气质。领口和袖口处都是修身的设计,可以勾勒出女性优美的身材曲线。<UNK>的裙摆,可以修饰腿部线条
,使腿部看起来更加修长。']
输入提示语：再见
['再见 再见,你曾经熟悉和热爱的<UNK>,再见,你曾经熟悉和热爱的<UNK>,再见,你曾经熟悉和热爱的<UNK>。']
```

For lora:

```
bash scripts/finetune_adgen_lora.sh
python inference_adgen.py --mode inference --skip-init --fp16 --ckpt_path checkpoints/finetune-chatglm-6b-adgen-04-03-06-48/
```

```
输入提示语：类型#上衣*材质#牛仔布*颜色#白色*风格#简约*图案#刺绣*衣样式#外套*衣款式#破洞
['类型#上衣*材质#牛仔布*颜色#白色*风格#简约*图案#刺绣*衣样式#外套*衣款式#破洞 牛仔外套以简约的白色为主色调,用
破洞和刺绣图案为外套增添时尚个性,打破单调感。外套上采用细腻的白色刺绣,点缀出细节美感,尽显精致工艺,上身效果十分
出众。衣身两侧破洞设计,打破整体简约的格调,增添时尚感,']
输入提示语：你好
['你好 你好👋!我是人工智能助手 ChatGLM-6B,很高兴见到你,欢迎问我任何问题。']
输入提示语：北京市是什么
['北京市是什么 北京市是中国的首都,也是全国的政治、文化、科技和商业中心。北京市位于中国的北部,是中国的政治、经
济、文化中心之一。北京市有着悠久的历史和文化,是世界著名的历史古都之一。\n\n北京市下辖16个区,包括东城区、西城区
、朝阳区、海淀区、丰台区、石景山区、门头沟区、房山区、通州区、顺义区、昌平区、大兴区、怀柔区、平谷区、密云区和
延庆区。北京市的景点众多,包括长城、故宫、天安门广场、颐和园、圆明园、天坛等,吸引着众多国内外游客前来观光旅游。
']
输入提示语：类型#裙*材质#针织*颜色#纯色*风格#复古*风格#文艺*风格#简约*图案#格子*图案#纯色*图案#复古*裙型#背带
裙*裙长#连衣裙*裙领型#半高领
['类型#裙*材质#针织*颜色#纯色*风格#复古*风格#文艺*风格#简约*图案#格子*图案#纯色*图案#复古*裙型#背带裙*裙长#连
衣裙*裙领型#半高领 简约纯色的针织连衣裙,是气质小姐姐们凹造型的好选择。半高领的领型,修饰颈部线条,凸显颈部优美的
线条。纯色的格子花纹,为裙身增添一丝复古文艺的气息,修饰出你的优雅文艺气质。背带式的背带裙,为裙身增添了一丝青春
活力,展现出你的年轻活力。']
输入提示语：再见
['再见 再见,祝您一切顺利!']
```

It seems that lora forgets less about what it has learned than ptuning.

Here is the parameter transformation script (from huggingface/chatglm-6b to sat/chatglm-6b):

```
python transform_param.py
```