import cv2
import numpy as np

# 读取模糊的文字图像
image = cv2.imread('D:/realesrgan-ncnn-vulkan-20220424-windows/input3.jpg', cv2.IMREAD_GRAYSCALE)

# 应用高斯模糊去噪
denoised_image = cv2.GaussianBlur(image, (5, 5), 0)

# 创建锐化滤波器
kernel = np.array([[0, -1, 0],
                   [-1, 5,-1],
                   [0, -1, 0]])

# 应用锐化滤波器
sharpened_image = cv2.filter2D(denoised_image, -1, kernel)

# 保存修复后的图像
cv2.imwrite('D:/realesrgan-ncnn-vulkan-20220424-windows/sharpened_text_image.png', sharpened_image)

# 显示原始和修复后的图像
# cv2.imshow('Original Image', image)
# cv2.imshow('Sharpened Image', sharpened_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()