from flask import Flask, request, jsonify
from paddleocr import PaddleOCR
import requests
from io import BytesIO
from PIL import Image
from pdf2image import convert_from_bytes
import tempfile
import os
import uuid
import base64

app = Flask(__name__)
# app.config['DEBUG'] = True

# 初始化 PaddleOCR
ocr = PaddleOCR(use_angle_cls=False, lang="ch")


# 设置请求头
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
}


@app.route('/ocr', methods=['POST'])
def ocr_service():
    data = request.get_json()
    if not data or 'file' not in data or 'format' not in data or 'ocr_type' not in data:
        return jsonify({'error': 'File or format or ocr_type are required'}), 400

    file_base64 = data['file']
    file_ext = data['format'].lower()

    if not file_base64 or not file_ext:
        return jsonify({'error': 'Invalid file or file_type'}), 400

    # ocr_type有两种类型，当用0时暂定，当为1时用paddle方式其余则报错
    ocr_type = data['ocr_type']
    if ocr_type == 0:
        # 使用默认的OCR处理逻辑
        print('Using default OCR logic')
    elif ocr_type == 1:
        # 使用paddlepaddle进行OCR处理
        result = paddle_ocr(file_base64, file_ext)
        return result
    else:
        # 对于无效的ocr_type值返回错误
        return jsonify({'error': 'Invalid ocr_type, supported types are 0 and 1'}), 400



def paddle_ocr(file_base64, file_ext):
    try:
        content_list = []
        save_dir = tempfile.gettempdir()
        # 将 Base64 字符串转换为字节
        file = base64.b64decode(file_base64)

        if file_ext in ['pdf']:
            # 处理 PDF 文件
            pdf_bytes = BytesIO(file)

            # 将 PDF 文件的每一页转换为图像
            images = convert_from_bytes(pdf_bytes.read())

            for i, image in enumerate(images):
                # 保存图像到本地
                image_path = os.path.join(save_dir, f"page_{i + 1}_{uuid.uuid4().hex}.png")
                image.save(image_path)
                image_ocr(content_list, image_path)
        elif file_ext in ['jpg', 'jpeg', 'png']:
            # 处理图像文件
            image = Image.open(BytesIO(file))
            image_path = os.path.join(save_dir, f"page_{uuid.uuid4().hex}.{file_ext}")
            image.save(image_path)
            image_ocr(content_list, image_path)
        else:
            return jsonify({'error': 'Unsupported file type'}), 400

        # content_list = sort_top_left_to_bottom_right(content_list)
        content = '\n'.join(content_list)
        return content
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def image_ocr(content_list, image_path):
    try:
        # 进行 OCR 识别
        result = ocr.ocr(image_path, cls=False)
        if result and result[0] is not None:
            for res in result:
                for line in res:
                    content_list.append(line[-1][0])
    finally:
        # 删除图像文件
        os.remove(image_path)


if __name__ == '__main__':
    # app.run(debug=True)
    app.run(host='0.0.0.0', port=5000)