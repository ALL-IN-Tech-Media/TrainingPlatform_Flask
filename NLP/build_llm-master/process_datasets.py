import os
import argparse
import json

def convert_cnews_file(input_file, output_file, mode="plain"):
    """
    Convert CNews from raw `label \t text` to plain text, prompt, or Qwen2.5 instruction-tuning JSONL format.

    Args:
        input_file (str): Path to raw CNews file
        output_file (str): Where to save processed output
        mode (str): 'plain', 'prompt', or 'qwen'
    """
    with open(input_file, 'r', encoding='utf-8') as fin, open(output_file, 'w', encoding='utf-8') as fout:
        for line in fin:
            if not line.strip():
                continue
            try:
                label, content = line.strip().split('\t', 1)
            except ValueError:
                continue  # skip broken lines

            if mode == "plain":
                fout.write(content.strip() + "\n")
            elif mode == "prompt":
                prompt = f"标签：{label}\n新闻内容：{content.strip()}\n\n"
                fout.write(prompt)
            elif mode == "qwen":
                messages = [
                    {"role": "system", "content": "请判断以下文章属于哪个类别："},
                    {"role": "user", "content": content.strip()},
                    {"role": "assistant", "content": label.strip()}
                ]
                fout.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
            else:
                raise ValueError("Unsupported mode: choose from 'plain', 'prompt', or 'qwen'")

    print(f"✅ Converted file saved to: {output_file}")

def convert_alpaca_data():
    input_path = "/home/ooin/lzz/study/build_llm/datasets/study_data/alpaca_data_zh_51k.json"
    output_path = "/home/ooin/lzz/study/build_llm/datasets/study_data/qwen_alpaca_data_zh_51k.json"

    with open(input_path, "r", encoding="utf-8") as fin:
        data = json.load(fin)

    with open(output_path, "w", encoding="utf-8") as fout:
        for item in data:
            # 拼接 user prompt
            if item.get("input", "").strip():
                user_content = f"{item['instruction'].strip()}\n{item['input'].strip()}"
            else:
                user_content = item["instruction"].strip()
            messages = [
                {"role": "system", "content": "你是一个乐于助人的助手。"},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": item["output"].strip()}
            ]
            fout.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")

    print(f"✅ 已保存到: {output_path}")

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Convert CNews format to training-ready format.")
    # parser.add_argument("--input", required=True, help="Path to raw CNews file" , default="/datasets/cnews/cnews.train.txt")
    # parser.add_argument("--output", required=True, help="Output file path" , default="/datasets/cnews_lang/train.jsonl")
    # parser.add_argument("--mode", choices=["plain", "prompt", "qwen"], default="plain", help="Output format")

    # args = parser.parse_args()
    # convert_cnews_file("datasets/cnews/cnews.train.txt", "datasets/cnews/train.jsonl", "qwen")
    # convert_cnews_file("datasets/cnews/cnews.val.txt", "datasets/cnews/val.jsonl", "qwen")
    # convert_cnews_file("datasets/cnews/cnews.test.txt", "datasets/cnews/test.jsonl", "qwen")
    convert_alpaca_data()
