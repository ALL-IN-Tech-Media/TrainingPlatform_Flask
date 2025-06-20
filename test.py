# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Example Python client for OpenAI Chat Completion using vLLM API server
NOTE: start a supported chat completion model server with `vllm serve`, e.g.
    vllm serve meta-llama/Llama-2-7b-chat-hf
"""

import argparse
import random
from openai import OpenAI

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "https://u651253-b924-3d564f20.bjc1.seetacloud.com:8443/v1"

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Who won the world series in 2020?"},
    {
        "role": "assistant",
        "content": "The Los Angeles Dodgers won the World Series in 2020.",
    },
    {"role": "user", "content": "Where was it played?"},
]

all_messages = [
    [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Who won the world series in 2020?"},
        {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
        {"role": "user", "content": "Where was it played?"},
    ],
    [
        {"role": "system", "content": "You are a knowledgeable sports expert."},
        {"role": "user", "content": "Who won the NBA finals in 2019?"},
        {"role": "assistant", "content": "The Toronto Raptors won the NBA Finals in 2019."},
        {"role": "user", "content": "Who was the MVP?"},
    ],
    [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Who won the FIFA World Cup in 2018?"},
        {"role": "assistant", "content": "France won the FIFA World Cup in 2018."},
        {"role": "user", "content": "Where was the final held?"},
    ],
    [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Who won the Super Bowl in 2021?"},
        {"role": "assistant", "content": "The Tampa Bay Buccaneers won the Super Bowl in 2021."},
        {"role": "user", "content": "Who was the quarterback?"},
    ],
    [
        {"role": "system", "content": "You are a tech expert assistant."},
        {"role": "user", "content": "Who invented the World Wide Web?"},
        {"role": "assistant", "content": "Tim Berners-Lee invented the World Wide Web in 1989."},
        {"role": "user", "content": "Where was he working at the time?"},
    ],
    [
        {"role": "system", "content": "You are a history teacher."},
        {"role": "user", "content": "When did the Berlin Wall fall?"},
        {"role": "assistant", "content": "The Berlin Wall fell in 1989."},
        {"role": "user", "content": "What was the main reason for its fall?"},
    ],
    [
        {"role": "system", "content": "You are a movie buff assistant."},
        {"role": "user", "content": "Who directed the movie Inception?"},
        {"role": "assistant", "content": "Christopher Nolan directed Inception."},
        {"role": "user", "content": "What year was it released?"},
    ],
    [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of Australia?"},
        {"role": "assistant", "content": "The capital of Australia is Canberra."},
        {"role": "user", "content": "Is it the largest city in Australia?"},
    ],
    [
        {"role": "system", "content": "You are a health advisor."},
        {"role": "user", "content": "What vitamin is produced when skin is exposed to sunlight?"},
        {"role": "assistant", "content": "Vitamin D is produced when skin is exposed to sunlight."},
        {"role": "user", "content": "Why is vitamin D important?"},
    ],
    [
        {"role": "system", "content": "You are a travel assistant."},
        {"role": "user", "content": "What is the tallest mountain in the world?"},
        {"role": "assistant", "content": "Mount Everest is the tallest mountain in the world."},
        {"role": "user", "content": "In which country is it located?"},
    ],
    # 可以继续添加更多不同的对话
]


def parse_args():
    parser = argparse.ArgumentParser(description="Client for vLLM API server")
    parser.add_argument(
        "--stream", action="store_true", help="Enable streaming response"
    )
    return parser.parse_args()


def main_test():
    client = OpenAI(
        # defaults to os.environ.get("OPENAI_API_KEY")
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    models = client.models.list()
    model = models.data[0].id

    # Chat Completion API
    chat_completion = client.chat.completions.create(
        messages=random.choice(all_messages),
        model=model,
        stream=False,
    )

    print("-" * 50)
    print("Chat completion results:")
    if False:
        for c in chat_completion:
            print(c.choices[0].message.content)
    else:
        print(chat_completion.choices[0].message.content)
    print("-" * 50)
    return chat_completion.choices[0].message.content


# if __name__ == "__main__":
#     args = parse_args()
#     main(args)
