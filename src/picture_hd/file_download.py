import os
import requests
from urllib.parse import urlparse

def download_file(pdf_url, save_dir=None):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    # 发送 HTTP 请求
    response = requests.get(pdf_url, stream=True, headers=headers, allow_redirects=True)
    downloaded_file = None

    # 打印最终的 URL
    print(f"Final URL: {response.url}")

    # 检查 HTTP 响应代码
    if response.status_code == 200:
        # 从 URL 中提取文件名
        disposition = response.headers.get('Content-Disposition')
        file_name = ""
        if disposition:
            # 从 Content-Disposition 中提取文件名
            index = disposition.find('filename=')
            if index > 0:
                file_name = disposition[index + 9:].strip('\"')
        else:
            # 从 URL 路径中提取文件名
            parsed_url = urlparse(pdf_url)
            file_name = os.path.basename(parsed_url.path)

        # 将文件名的后缀名转换为小写
        file_name = convert_file_extension_to_lowercase(file_name)

        # 保存文件的路径
        if not save_dir:
            save_dir = os.getenv('TMPDIR', '/tmp')
        save_file_path = os.path.join(save_dir, file_name)

        # 将输入流写入文件
        with open(save_file_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    file.write(chunk)

        downloaded_file = save_file_path
        print(f"File downloaded: {downloaded_file}")
    else:
        print(f"No file to download. Server replied HTTP code: {response.status_code}")

    return downloaded_file

def convert_file_extension_to_lowercase(file_name):
    name, extension = os.path.splitext(file_name)
    return f"{name}{extension.lower()}"

# 示例用法
if __name__ == "__main__":
    pdf_url = "https://www.yindeng.com.cn/resource/54707/54723/54740/925994/926526/990694/1366436/16475078056081803466137.pdf"
    save_dir = r"C:\Users\haojie.liu\Downloads"
    downloaded_file = download_file(pdf_url, save_dir)