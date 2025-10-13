import requests
from bs4 import BeautifulSoup
import os
from datetime import datetime

class PDFInfo:
    def __init__(self, title, href, publish_time, pdf_name=None, pdf_url=None):
        self.title = title
        self.href = href
        self.publish_time = publish_time
        self.pdf_name = pdf_name
        self.pdf_url = pdf_url

    def __repr__(self):
        return f"PDFInfo(title={self.title}, href={self.href}, publish_time={self.publish_time}, pdf_name={self.pdf_name}, pdf_url={self.pdf_url})"

def download_pdf(pdf_url, save_dir):
    response = requests.get(pdf_url, headers=headers, allow_redirects=True)
    if response.status_code == 200:
        file_name = os.path.join(save_dir, pdf_url.split('/')[-1])
        with open(file_name, 'wb') as f:
            f.write(response.content)
        return file_name
    else:
        raise Exception(f"Failed to download PDF: {pdf_url}")

# 设置请求头
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
}

def main():
    pdf_name_list = []
    current_page = 102
    save_dir = "D:\\"

    while True:
        retire_url = f"https://www.yindeng.com.cn/Home/cn/zczrxx/xxpl/zrgg/InfoList_{current_page}.shtml"
        charge_url = f"https://www.yindeng.com.cn/Home/cn/zczrxx/xxpl/zrjggg/InfoList_{current_page}.shtml"
        # baidu_url = f"https://www.baidu.com"

        response = requests.get(charge_url, headers=headers, allow_redirects=True)
        if response.status_code == 404:
            break

        soup = BeautifulSoup(response.content, 'html.parser')
        new_list_element = soup.find(class_='newlist')
        ul_element = new_list_element.find('ul')
        li_elements = ul_element.find_all('li')

        pdf_info_list = []
        for li_element in li_elements:
            a_element = li_element.find('a')
            href = a_element['href']
            title = a_element['title']
            if title in pdf_name_list:
                continue

            time_element = li_element.find_all('span')[1]
            time = time_element.get_text()
            publish_time = datetime.strptime(time, '%Y-%m-%d')

            pdf_info = PDFInfo(title=title, href=href, publish_time=publish_time)
            pdf_info_list.append(pdf_info)

        for pdf_info in pdf_info_list:
            response = requests.get(pdf_info.href, headers=headers, allow_redirects=True)
            soup = BeautifulSoup(response.content, 'html.parser')
            pdf_elements = soup.find_all('a', href=lambda href: href and href.endswith('.pdf'))
            for pdf_element in pdf_elements:
                pdf_url = pdf_element['href']
                pdf_info.pdf_url = pdf_url
                try:
                    file_name = download_pdf(pdf_url, save_dir)
                    pdf_info.pdf_name = file_name
                    print(f"pdfInfo: {pdf_info}")
                except Exception as e:
                    print(f"Error downloading PDF: {e}")

        current_page += 1

if __name__ == "__main__":
    main()