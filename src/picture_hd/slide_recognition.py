import time
import cv2
import numpy as np
from selenium import webdriver
from selenium.webdriver import ActionChains

def get_image(driver, xpath):
    img_element = driver.find_element_by_xpath(xpath)
    img_base64 = driver.execute_script("return arguments[0].toDataURL('image/png').substring(22);", img_element)
    img_data = base64.b64decode(img_base64)
    img_array = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return img

def calculate_distance(bg_img, slider_img):
    result = cv2.matchTemplate(bg_img, slider_img, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    return max_loc[0]

def generate_track(distance):
    track = []
    current = 0
    mid = distance * 3 / 5
    t = 0.2
    v = 0
    while current < distance:
        if current < mid:
            a = 2
        else:
            a = -3
        v0 = v
        s = v0 * t + 0.5 * a * (t ** 2)
        current += s
        track.append(round(s))
        v = v0 + a * t
    return track

def slide(driver, slider, track):
    ActionChains(driver).click_and_hold(slider).perform()
    for x in track:
        ActionChains(driver).move_by_offset(xoffset=x, yoffset=0).perform()
    time.sleep(0.5)
    ActionChains(driver).release().perform()

driver = webdriver.Chrome()
driver.get('URL_OF_YOUR_PAGE')

bg_img = get_image(driver, 'XPATH_OF_BACKGROUND_IMAGE')
slider_img = get_image(driver, 'XPATH_OF_SLIDER_IMAGE')

distance = calculate_distance(bg_img, slider_img)
track = generate_track(distance)

slider = driver.find_element_by_xpath('XPATH_OF_SLIDER')
slide(driver, slider, track)