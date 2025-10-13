import requests
import datetime

from src.gpt.llm_key import azure_api_key
from src.gpt.promt import fewshot, questionList

messages = [{
    "role": "system",

    "content": "你是一个中国招商银行的贷款部门电话客服人员，客户通过电话咨询贷款相关信息。你准备好了吗？"
}]

url = "https://chatgpt-3-5.openai.azure.com/openai/deployments/gpt-35-turbo/chat/completions?api-version=2023-07-01-preview"
header = {"api-key": azure_api_key}
with requests.session() as session:
    r1 = session.post(url, headers=header, verify=False, json={"messages": messages})
    r2 = session.post("https://baidu.com", headers=header, verify=False, json={"messages": messages})

    print(r1.json().get("choices", [])[0].get("message", {}).get("content"))

    total_delta = 0
    loop_cnt = len(questionList)
    for i in range(loop_cnt):
        message_text = [{
            "role": "system",
            "content": fewshot + questionList[i]
        }]

        begin_time = datetime.datetime.now()
        r2 = session.post(url, headers=header, verify=False, json={"messages": message_text})
        end_time = datetime.datetime.now()

        print("****** 问题：" + questionList[i])
        print("====== 回答：" + r2.json().get("choices", [])[0].get("message", {}).get("content"))
        delta = end_time - begin_time
        milliseconds = delta.total_seconds() * 1000
        total_delta = total_delta + milliseconds
        print("本次调用gpt-3.5延时（毫秒）：", milliseconds)
        print("-----------------------------------------------------------------")

    print("通用客服回答：" + str(loop_cnt) + "次调用gpt-3.5的平均延时（毫秒）：", total_delta / loop_cnt)
