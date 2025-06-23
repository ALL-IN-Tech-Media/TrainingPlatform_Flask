import torch
import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from transformers import TrainingArguments
from transformers import Trainer
from transformers import DataCollatorForLanguageModeling
import os
import json

# https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct

# 明确指定使用单卡
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"  # 只使用第一张卡
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_dataset(tokenizer):
    """加载短信垃圾邮件分类数据集"""
    # 读取TSV文件
    df = pd.read_csv('./datasets/sms+spam+collection/SMSSpamCollection.tsv', 
                     sep='\t', header=None, names=['label', 'text'])
    
    result_data = []
    for _, row in df.iterrows():
        message = [
            {'role': 'system', 'content': 'You are a spam detection expert. Please determine if the following SMS message is spam or ham (legitimate). Only respond with "spam" or "ham".'},
            {'role': 'user', 'content': row['text']},
            {'role': 'assistant', 'content': row['label']}
        ]
        inputs = tokenizer.apply_chat_template(message, add_generation_prompt=False, tokenize=True)
        result_data.append(inputs)

    return result_data

def get_dataset_with_split(tokenizer, split_ratio=0.8):
    """加载并分割数据集"""
    # 读取TSV文件
    df = pd.read_csv('./datasets/sms+spam+collection/SMSSpamCollection.tsv', 
                     sep='\t', header=None, names=['label', 'text'])
    
    # 分割数据集
    train_size = int(len(df) * split_ratio)
    train_df = df[:train_size]
    val_df = df[train_size:]
    
    print(f"训练集大小: {len(train_df)}")
    print(f"验证集大小: {len(val_df)}")
    
    # 处理训练集
    train_data = []
    for _, row in train_df.iterrows():
        message = [
            {'role': 'system', 'content': 'You are a spam detection expert. Please determine if the following SMS message is spam or ham (legitimate). Only respond with "spam" or "ham".'},
            {'role': 'user', 'content': row['text']},
            {'role': 'assistant', 'content': row['label']}
        ]
        inputs = tokenizer.apply_chat_template(message, add_generation_prompt=False, tokenize=True)
        train_data.append(inputs)
    
    # 处理验证集
    val_data = []
    for _, row in val_df.iterrows():
        message = [
            {'role': 'system', 'content': 'You are a spam detection expert. Please determine if the following SMS message is spam or ham (legitimate). Only respond with "spam" or "ham".'},
            {'role': 'user', 'content': row['text']},
            {'role': 'assistant', 'content': row['label']}
        ]
        inputs = tokenizer.apply_chat_template(message, add_generation_prompt=False, tokenize=True)
        val_data.append(inputs)
    
    return train_data, val_data

def get_dataset_from_jsonl(tokenizer, path):
    """根据新的jsonl格式加载数据集，并将prompt拆分为system和user"""
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            prompt = item["prompt"]
            answer = item["answer"]
            # 拆分prompt
            if '\n' in prompt:
                system_prompt, user_content = prompt.split('\n', 1)
                system_prompt = system_prompt.strip()
                user_content = user_content.strip()
            else:
                system_prompt = prompt.strip()
                user_content = ''
            message = [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_content},
                {'role': 'assistant', 'content': answer}
            ]
            inputs = tokenizer.apply_chat_template(message, add_generation_prompt=False, tokenize=True)
            data.append(inputs)
    return data

# watch -n 1 nvidia-smi
def demo():
    print("=== 短信垃圾邮件分类微调 ===")
    print(f"使用设备: {device}")
    print(f"CUDA设备数量: {torch.cuda.device_count()}")
    
    # 加载模型和分词器
    model_name = '/home/ooin/lzz/study/build_llm/huggingface/Qwen2.5-0.5B-Instruct'
    estimator = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # 设置pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        estimator.config.pad_token_id = tokenizer.eos_token_id
    
    print(f"模型加载完成: {model_name}")
    
    # 获取数据集（新格式）
    train_data = get_dataset_from_jsonl(tokenizer, './datasets/cnews/train.jsonl')
    val_data = get_dataset_from_jsonl(tokenizer, './datasets/cnews/val.jsonl')
    
    # 训练参数 - 明确指定单卡
    arguments = TrainingArguments(
        output_dir='Qwen2.5-0.5B-Instruct-SMS-Spam',
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        optim='adamw_torch',
        num_train_epochs=3,
        learning_rate=2e-5,
        eval_strategy='epoch',  # 每个epoch评估一次
        save_strategy='epoch',
        logging_strategy='steps',
        logging_steps=50,
        gradient_accumulation_steps=4,
        save_total_limit=3,
        load_best_model_at_end=True,  # 加载最佳模型
        metric_for_best_model='eval_loss',
        greater_is_better=False,
        warmup_steps=100,
        weight_decay=0.01,
        fp16=True,  # 使用混合精度训练
        dataloader_pin_memory=False,
        # 明确指定单卡
        # no_cuda=False,
        # local_rank=-1,  # 禁用分布式训练
    )

    # 创建数据整理器
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=False,
        pad_to_multiple_of=8  # 优化GPU内存使用
    )
    
    # 创建训练器
    trainer = Trainer(
        model=estimator,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=arguments,
        data_collator=data_collator,
    )

    print("开始训练...")
    trainer.train()
    
    # 保存最终模型
    print("保存模型...")
    trainer.save_model('./huggingface/Qwen2.5-0.5B-Instruct-SMS-Spam')
    tokenizer.save_pretrained('./huggingface/Qwen2.5-0.5B-Instruct-SMS-Spam')
    
    print("训练完成！模型已保存到: ./huggingface/Qwen2.5-0.5B-Instruct-SMS-Spam")

def test_model():
    """测试微调后的模型"""
    print("=== 测试微调后的模型 ===")
    
    # 加载微调后的模型
    model_path = './huggingface/Qwen2.5-0.5B-Instruct-SMS-Spam'
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # 测试例子
    test_cases = [
        {
            "text": "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121",
            "expected": "spam"
        },
        {
            "text": "Ok lar... Joking wif u oni...",
            "expected": "ham"
        },
        {
            "text": "URGENT! You have won a 1 week FREE membership in our £100,000 Prize Jackpot!",
            "expected": "spam"
        },
        {
            "text": "I'm gonna be home soon and i don't want to talk about this stuff anymore tonight",
            "expected": "ham"
        }
    ]
    
    correct = 0
    total = len(test_cases)
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n测试 {i}:")
        print(f"短信: {case['text']}")
        print(f"期望: {case['expected']}")
        
        # 构建输入
        message = [
            {'role': 'system', 'content': 'You are a spam detection expert. Please determine if the following SMS message is spam or ham (legitimate). Only respond with "spam" or "ham".'},
            {'role': 'user', 'content': case['text']}
        ]
        
        inputs = tokenizer.apply_chat_template(message, add_generation_prompt=True, return_tensors="pt").to(device)
        
        # 生成响应
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=10,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.2,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 提取生成的文本
        generated_text = response.split("assistant")[-1].strip()
        print(f"生成: {generated_text}")
        
        # 判断结果
        if case['expected'] in generated_text.lower():
            correct += 1
            print("✓ 正确")
        else:
            print("✗ 错误")
    
    print(f"\n准确率: {correct}/{total} ({correct/total*100:.2f}%)")

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        test_model()
    else:
        demo()