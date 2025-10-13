import requests
import json
import logging
import os

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

KNOWLEDGE_BASE_URL = "https://www.cybotstar.cn/openapi/v1/knowledge/file/"


class RybotAuthConfig:
    def __init__(self):
        self.rybot_auth = {
            "split_method_meta": "{\"length\": 300}",
            "split_method": "intelligence"
        }

    def get_rybot_auth(self):
        return self.rybot_auth


rybot_auth_config = RybotAuthConfig()


def set_rybot_auth_info(headers):
    # 设置认证信息
    headers['Authorization'] = 'Bearer your_token'


def execute_upload_post(import_file, knowledge_base_id):
    url = KNOWLEDGE_BASE_URL
    file_name = os.path.basename(import_file)
    files = {
        'file': (import_file, open(import_file, 'rb'), 'application/octet-stream'),
        'knowledge_base_id': (None, knowledge_base_id),
        'split_method_meta': (None, rybot_auth_config.get_rybot_auth()['split_method_meta'], 'application/json'),
        'split_method': (None, rybot_auth_config.get_rybot_auth()['split_method']),
        'file_name': (None, file_name)
    }
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'cybertron-robot-key':'U65TTweBiZW%2F6eF8yJfZpLTpUic%3D',
        'cybertron-robot-token':'MTcyODk4NTMyMTc5OApBSWhTTHZoaC9yNDY5cy9DTWc1UzI1NU16TmM9',
        'username':'haojie.liu@brgroup.com'
    }
    set_rybot_auth_info(headers)

    response = requests.post(url, files=files, headers=headers)

    if response.status_code == 200:
        response_data = response.json()
        logger.info("responseString: %s", response.text)
        if response_data.get('code') == '000000':
            data = response_data.get('data')
            if isinstance(data, dict):
                return data.get('id')
    else:
        logger.error("HTTP request failed with status code %d", response.status_code)

    return None


# 示例用法
if __name__ == "__main__":
    import_file = r"C:\Users\haojie.liu\Downloads\00054c5b5e854131ac3f2a45b2057348.pdf"
    #with open(import_file, 'wb') as file:
        #file.read()
    knowledge_base_id = "1399"
    result_id = execute_upload_post(import_file, knowledge_base_id)
    if result_id:
        print(f"Uploaded file ID: {result_id}")
    else:
        print("Failed to upload file")