# README

## 安装

```
pytorch
transformers
trl
```

## 运行

```
python train_pair_reward.py \
    --model_name_or_path /data/share/llama3.1-8b-base/ \
    --dataset_name data/pair_dataset \
    --output_dir llama3.1_reward-lora \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 1 \
    --gradient_checkpointing True \
    --bf16 True \
    --attn_implementation flash_attention_2 \
    --learning_rate 1.0e-4 \
    --logging_steps 25 \
    --eval_strategy steps \
    --eval_steps 50 \
    --max_length 2048 \
    --use_peft \
    --lora_r 4 \
    --lora_alpha 8
```

**重要调节参数：**

- lora_r 指的是 lora 的 rank，需要细调。如果过拟合严重就调小，欠拟合就调大。另外， 一般 lora_alpha=2*lora_r需要跟着调整。

## 数据格式

数据存在 ``data/pair_dataset/`` 下，分为训练集和测试集（对应 train.pkl, test.pkl），每个数据集具有相同的格式，具体如下：

```
[
{'chosen': 'xxx', 'rejected': 'yyy'},
{'chosen': 'aaa', 'rejected': 'bbb'},
...
]
```

- 文件保存为了 pkl 格式，当前文件夹下有一个符合格式的数据集例子，可以参考。
- 整体是一个 list
- 每个list的元素是一个dict，有chosen和rejected两个key，分别对应好的回答和不好的回答

**构造数据的注意事项：**

- 无论是 chosen 还是 rejected，都应当为原始的问题 + 回复（例如："[USER PROMPT]\n" + prompt + "\n[ASSISTANT ANSWER]\n" + answer）
- 训练集和测试集的问题一定要严格区分。例如，应当用训练集问题前80%的数据构造若干回答，并构成好坏pair，作为训练集；用后 20% 的数据构造若干回答，构造好坏 pair。换言之，训练、测试集的**问题**一定要是不同的。

