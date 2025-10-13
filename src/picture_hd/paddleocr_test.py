from paddleocr import PaddleOCR

image2 = r"D:/realesrgan-ncnn-vulkan-20220424-windows/input4.jpg"
ocr = PaddleOCR(use_angle_cls=False, lang="ch")
result = ocr.ocr(image2, cls=False)
content_list=[]
for idx in range(len(result)):
    res = result[idx]
    for line in res:
        content_list.append(line[-1][0])
content='\n'.join(content_list)
print(content)