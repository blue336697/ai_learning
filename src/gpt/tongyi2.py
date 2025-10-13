import dashscope
import datetime

from src.gpt.llm_key import dashscope_api_key
from src.gpt.promt import questionList, get_messages

dashscope.api_key = dashscope_api_key


def test_completions_with_params(question):
    begin_time = datetime.datetime.now()
    resp = dashscope.Generation.call(
        dashscope.Generation.Models.qwen_turbo,
        messages=get_messages(question),
        result_format='message',
    )
    end_time = datetime.datetime.now()

    content = resp.output.choices[0]['message']['content']

    print("input: %s" % (question))
    print("output: %s" % (content))
    print("非流式调用：tongyi2总延时（毫秒）: %s\n" % ((end_time - begin_time).total_seconds() * 1000))


def test_stream_completions(question):
    begin_time = datetime.datetime.now()
    resp = dashscope.Generation.call(
        dashscope.Generation.Models.qwen_turbo,
        messages=get_messages(question),
        result_format='message',
        stream=True,
        incremental_output=True
    )

    first = True
    collected_messages = []
    for result in resp:
        chunk_time = datetime.datetime.now()
        chunk_message = result.output.choices[0]['message']['content']
        collected_messages.append(chunk_message)
        if first and '，' in chunk_message:
            first = False
            collected_messages.append(
                f"<本次调用tongyi2_stream首句延时：%s毫秒>" % ((chunk_time - begin_time).total_seconds() * 1000))

    collected_messages = [m for m in collected_messages if m is not None]
    content = ''.join([m for m in collected_messages])

    end_time = datetime.datetime.now()

    print("input: %s" % (question))
    print("output: %s" % (content))
    print("流式调用：tongyi2_stream总延时：%s毫秒 \n" % ((end_time - begin_time).total_seconds() * 1000))


loop_cnt = len(questionList)

print("------------------------通义千问-结果-----------------------------------------")
for i in range(loop_cnt):
    print("----------------------------------------")
    test_completions_with_params(questionList[i])
    test_stream_completions(questionList[i])
