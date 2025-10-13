import openai  # for openai 0.*
import datetime

from src.gpt.llm_key import azure_api_key
from src.gpt.promt import fewshot, questionList, get_messages

openai.api_type = "azure"
openai.api_base = "https://chatgpt-3-5.openai.azure.com"
openai.api_version = "2023-05-15"
openai.api_key = azure_api_key


def test_completions_with_params(question):

    begin_time = datetime.datetime.now()
    response = openai.ChatCompletion.create(
        engine="gpt-35-turbo",
        messages=get_messages(question),
        temperature=0,
        # max_tokens=50,
        # top_p=0.95,
        # frequency_penalty=0,
        # presence_penalty=0,
        # stop=None,
        stream=False
    )

    end_time = datetime.datetime.now()

    content = response.choices[0].message.content

    print("input: %s" % (question))
    print("output: %s" % (content))
    print("非流式调用：azure总延时（毫秒）: %s\n" % ((end_time - begin_time).total_seconds() * 1000))


def test_stream_completions(question):

    begin_time = datetime.datetime.now()

    response = openai.ChatCompletion.create(
        engine="gpt-35-turbo",
        messages=get_messages(question),
        temperature=0,
        # max_tokens=350,
        # top_p=0.95,
        # frequency_penalty=0,
        # presence_penalty=0,
        # stop=None,
        stream=True
    )

    first = True
    collected_messages = []
    for chunk in response:
        chunk_time = datetime.datetime.now()
        chunk_message = chunk.choices[0].delta.content
        collected_messages.append(chunk_message)
        # print(f"Message received {(chunk_time-begin_time).total_seconds() * 1000:.2f} 毫秒 after request: {chunk_message}")
        if first and chunk_message is not None and '，' in chunk_message:
            first = False
            collected_messages.append(
                f"<本次调用azure_stream首句延时：%s毫秒>" % ((chunk_time - begin_time).total_seconds() * 1000))

    collected_messages = [m for m in collected_messages if m is not None]
    content = ''.join([m for m in collected_messages])

    end_time = datetime.datetime.now()

    print("input: %s" % (question))
    print("output: %s" % (content))
    print("流式调用：azure_stream总延时：%s毫秒 \n" % ((end_time - begin_time).total_seconds() * 1000))


loop_cnt = len(questionList)

print("------------------------azure-结果-----------------------------------------")
for i in range(loop_cnt):
    print("-----------------------------------------")
    test_completions_with_params(questionList[i])
    test_stream_completions(questionList[i])
