import os
import sys
import platform
import signal
from llmtuner import ChatModel
from .transforms import val_transforms


os_name = platform.system()
clear_command = 'cls' if os_name == 'Windows' else 'clear'
stop_stream = False

def build_prompt(history, mode):
    prompt = "欢迎使用 OPHGLM 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序"
    idx = 0
    for query, response in history:
        if mode == 1:
            if idx > 0:
                prompt += f"\n\n用户：{query}"
                prompt += f"\n\nOPHGLM：{response}"
            else:
                prompt += f"\n\n用户：{'请对用户所提供的眼底图像进行智能诊断，并给出诊断结果。'}"
                prompt += f"\n\nOPHGLM：{response}"
        else:
            prompt += f"\n\n用户：{query}"
            prompt += f"\n\nOPHGLM：{response}"

        idx += 1
    return prompt


def signal_handler(signal, frame):
    global stop_stream
    stop_stream = True


def main():
    chat_model = ChatModel()
    history = []

    global stop_stream
    print("欢迎使用 OPHGLM 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序")
    while True:
        history = []

        imgpath = input('\n请输入眼底图像路径或直接开始聊天：')

        if imgpath == 'stop':
            break
        if imgpath == 'clear':
            os.system(clear_command)
            history = []
            print("欢迎使用 OPHGLM 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序")
            continue

        if len(imgpath) > 0 and 'jpg' in imgpath:
            query = val_transforms(imgpath)
            mode = 1
        elif len(imgpath) > 0:
            query = imgpath
            mode = 2
            
        while True:
        
            if query.strip() == "stop":
                sys.exit(0)
            if query.strip() == "clear":
                history = []
                os.system(clear_command)
                print("欢迎使用 OPHGLM 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序")
                break

            print("OPHGLM: ", end="", flush=True)
            count = 0
            response = ""
            for new_text in chat_model.stream_chat(query, history):
                response += new_text
                print(new_text, end="", flush=True)
            history = history + [(query, response)]


            os.system(clear_command)
            print(build_prompt(history, mode), flush=True)

            query = input("\n\n用户：")


if __name__ == "__main__":
    main()
