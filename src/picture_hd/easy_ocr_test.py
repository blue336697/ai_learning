import easyocr
from flask import Flask, request, jsonify
import requests
from io import BytesIO
from pdf2image import convert_from_bytes
import os

app = Flask(__name__)
# app.config['DEBUG'] = True

# 设置请求头
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
}

reader = easyocr.Reader(['ch_sim','en'], gpu=False) # this needs to run only once to load the model into memory

@app.route('/ocr', methods=['POST'])
def ocr_service():
    data = request.json
    image_url = data.get('url')
    if not image_url:
        return jsonify({'error': 'URL is required'}), 400

    try:
        # # 远程下载PDF识别
        response = requests.get(image_url, headers=headers, allow_redirects=True)
        response.raise_for_status()

        pdf_bytes = BytesIO(response.content)

        # 将 PDF 文件的每一页转换为图像
        images = convert_from_bytes(pdf_bytes.read())

        # 进行 OCR 识别并保存图像
        content_list = []
        save_dir = "D:\\"
        for i, image in enumerate(images):
            # 保存图像到本地
            image_path = os.path.join(save_dir, f"page_{i + 1}.png")
            image.save(image_path, 'PNG')

            # 进行 OCR 识别
            result = reader.readtext(image_path, detail = 0)
            for res in result:
                content_list.append(res)
        content = '\n'.join(content_list)
        return jsonify({'text': content})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # app.debug = True
    app.run(host='0.0.0.0', port=5000)