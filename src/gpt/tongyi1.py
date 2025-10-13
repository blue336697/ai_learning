import broadscope_bailian
import datetime

from src.gpt.llm_key import access_key_id, access_key_secret, agent_key
from src.gpt.promt import fewshot, questionList, get_messages

# app_id = "c94b91639f304e6891c77a74c942c7a3" #Max
# app_id = "64f4c3afd7694a01839d27d1be0a630a" #Plus
app_id = "9f10c6b043a843889dbb00932a80a47d"  # Tubor


# app_id = "2a88098ee5ad4bba8844f6a8ccfd5485" #开源版


def get_token():
    client = broadscope_bailian.AccessTokenClient(access_key_id=access_key_id,
                                                  access_key_secret=access_key_secret,
                                                  agent_key=agent_key)
    return client.get_token()


token = get_token()


def test_completions_with_params(question):
    begin_time = datetime.datetime.now()
    resp = broadscope_bailian.Completions(token=token).create(
        app_id=app_id,
        messages=get_messages(question),
        # 设置模型参数topP的值
        top_p=0.2,
        # 设置模型参数topK
        top_k=50,
        # 设置模型参数seed
        seed=2222,
        # 设置模型参数temperature
        temperature=0.3,
        # 设置模型参数max tokens
        max_tokens=500,
        # 按message方式返回结果
        result_format="message",
        # 设置停止词
        # stop=["景点"]
    )
    end_time = datetime.datetime.now()

    content = resp.get("Data", {}).get("Choices", [])[0].get("Message", {}).get("Content")

    print("input: %s" % (question))
    print("output: %s" % (content))
    print("非流式调用：tongyi总延时（毫秒）: %s\n" % ((end_time - begin_time).total_seconds() * 1000))


def test_stream_completions(question):
    begin_time = datetime.datetime.now()
    resp = broadscope_bailian.Completions(token=token).create(
        app_id=app_id,
        messages=get_messages(question),
        stream=True,
        # 返回choice message结果
        result_format="message",
        # 开启增量输出模式，后面输出不会包含已经输出的内容
        incremental_output=True
    )

    first = True
    collected_messages = []
    for result in resp:
        chunk_time = datetime.datetime.now()
        chunk_message = result.get("Data", {}).get("Choices", [])[0].get("Message", {}).get("Content")
        collected_messages.append(chunk_message)
        if first and '，' in chunk_message:
            first = False
            collected_messages.append(
                f"<本次调用tongyi_stream首句延时：%s毫秒>" % ((chunk_time - begin_time).total_seconds() * 1000))

    collected_messages = [m for m in collected_messages if m is not None]
    content = ''.join([m for m in collected_messages])

    end_time = datetime.datetime.now()

    print("input: %s" % (question))
    print("output: %s" % (content))
    print("流式调用：tongyi_stream总延时：%s毫秒 \n" % ((end_time - begin_time).total_seconds() * 1000))


loop_cnt = len(questionList)

print("------------------------通义千问-结果-----------------------------------------")
for i in range(loop_cnt):
    print("----------------------------------------")
    test_completions_with_params(questionList[i])
    test_stream_completions(questionList[i])
