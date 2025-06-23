import requests

# 设置API地址（你的本地vllm服务）
API_URL = "http://localhost:9090/v1/chat/completions"

# 构建聊天消息历史（支持多轮）
chat_history = [
    {"role": "system", "content": "你是一个乐于助人的助手。"},
]

def chat(user_input):
    # 添加用户输入到历史
    chat_history.append({"role": "user", "content": user_input})

    # 构建请求体
    payload = {
        # "model": "Qwen2.5-1.5B-Instruct-finetuned",  # 可随意填写，vLLM只看部署的模型
        "messages": chat_history,
        "temperature": 0.7,
        "max_tokens": 512,
        "top_p": 0.95
    }

    # 发送请求
    response = requests.post(API_URL, json=payload)
    response.raise_for_status()

    # 解析模型回复
    message = response.json()["choices"][0]["message"]["content"]

    # 添加模型回复到历史
    chat_history.append({"role": "assistant", "content": message})

    return message

if __name__ == "__main__":
    print("🤖 已连接至本地 vLLM 模型，输入 'exit' 退出对话。\n")
    while True:
        user_input = input("你：")
        if user_input.lower() in ["exit", "quit", "退出"]:
            break
        reply = chat(user_input)
        print(f"助手：{reply}\n")
