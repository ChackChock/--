import os

import cv2
from matplotlib import pyplot as plt


eyes_data = cv2.CascadeClassifier("haarcascades/haarcascade_eye.xml")
mouth_data = cv2.CascadeClassifier("haarcascades/haarcascade_mcs_mouth.xml")

base_path = "part_1/images"

for filename in os.listdir(base_path):
    img = cv2.imread(os.path.join(base_path, filename))
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    eyes_coords = list(eyes_data.detectMultiScale(img_gray, minSize=(20, 20)))
    mouth_coords = list(mouth_data.detectMultiScale(img_gray, minSize=(20, 20)))

    for x, y, width, height in eyes_coords:
        center = (x + width // 2, y + height // 2)
        cv2.circle(img_rgb, center, max(width, height) // 2, (50, 255, 50), 5)

    for x, y, width, height in mouth_coords:
        right_bottom = (x + width, y + height)
        cv2.rectangle(img_rgb, (x, y), right_bottom, (255, 50, 50), 5)

    plt.subplot(1, 1, 1)
    plt.imshow(img_rgb)
    plt.show()
