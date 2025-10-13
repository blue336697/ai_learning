import os
import qianfan
import datetime
from src.gpt.llm_key import qianfan_access_key, qianfan_secret_key
from src.gpt.promt import questionList, get_messages, get_messages_wenxin

# 使用安全认证AK/SK鉴权，通过环境变量方式初始化；替换下列示例中参数，安全认证Access Key替换your_iam_ak，Secret Key替换your_iam_sk
os.environ["QIANFAN_ACCESS_KEY"] = qianfan_access_key
os.environ["QIANFAN_SECRET_KEY"] = qianfan_secret_key


def test_completions_with_params(question):
    begin_time = datetime.datetime.now()
    comp = qianfan.ChatCompletion()
    resp = comp.do(model="ERNIE-Speed",
                   messages=get_messages_wenxin(question),
                   temperature=0.1)
    end_time = datetime.datetime.now()

    content = resp.get("body", {}).get("result")

    print("input: %s" % (question))
    print("output: %s" % (content))
    print("非流式调用：wenxin总延时（毫秒）: %s\n" % ((end_time - begin_time).total_seconds() * 1000))


def test_stream_completions(question):
    begin_time = datetime.datetime.now()
    comp = qianfan.ChatCompletion()
    resp = comp.do(model="ERNIE-Speed",
                   messages=get_messages_wenxin(question),
                   temperature=0.1,
                   stream=True)

    first = True
    collected_messages = []
    for r in resp:
        chunk_time = datetime.datetime.now()
        chunk_message = r.get("result")
        collected_messages.append(chunk_message)
        if first and chunk_message is not None and '，' in chunk_message:
            first = False
            collected_messages.append(
                f"<本次调用wenxin_stream首句延时：%s毫秒>" % ((chunk_time - begin_time).total_seconds() * 1000))

    collected_messages = [m for m in collected_messages if m is not None]
    content = ''.join([m for m in collected_messages])

    end_time = datetime.datetime.now()

    print("input: %s" % (question))
    print("output: %s" % (content))
    print("流式调用：wenxin_stream总延时：%s毫秒 \n" % ((end_time - begin_time).total_seconds() * 1000))


loop_cnt = len(questionList)

print("------------------------文心一言-结果-----------------------------------------")
for i in range(loop_cnt):
    print("----------------------------------------")
    test_completions_with_params(questionList[i])
    test_stream_completions(questionList[i])
