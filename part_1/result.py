import cv2
from matplotlib import pyplot as plt

img = cv2.imread("Part1/face.jpg")

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

righteye_data = cv2.CascadeClassifier("haarcascades/haarcascade_righteye_2splits.xml")
lefteye_data = cv2.CascadeClassifier("haarcascades/haarcascade_lefteye_2splits.xml")
face_data = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_alt2.xml")
catface_data = cv2.CascadeClassifier(
    "haarcascades/haarcascade_frontalcatface_extended.xml"
)

righteye_coords = righteye_data.detectMultiScale(img_gray, minSize=(20, 20)).tolist()[0]
lefteye_coords = lefteye_data.detectMultiScale(img_gray, minSize=(20, 20)).tolist()[0]
face_coords = face_data.detectMultiScale(img_gray, minSize=(20, 20)).tolist()[0]
catface_coords = catface_data.detectMultiScale(img_gray, minSize=(20, 20)).tolist()[0]

print(f"{righteye_coords=}")
print(f"{lefteye_coords=}")
print(f"{face_coords=}")
print(f"{catface_coords=}")

with open("Part1/coords.txt", "w") as storage:
    storage.write(",".join(map(str, righteye_coords)) + "\n")
    storage.write(",".join(map(str, lefteye_coords)) + "\n")
    storage.write(",".join(map(str, face_coords)) + "\n")
    storage.write(",".join(map(str, catface_coords)) + "\n")

with open("Part1/coords.txt", "r") as storage:
    coords = storage.read().split("\n")[:-1]

    for i in range(len(coords)):
        coords[i] = coords[i].split(",")
        coords[i] = list((map(int, coords[i])))

    for i, (x, y, width, height) in enumerate(coords):
        if i in [0, 1]:
            cv2.circle(
                img_rgb,
                (x + (width // 2), y + (height // 2)),
                width // 2,
                (0, 255, 0),
                5,
            )
        else:
            cv2.rectangle(img_rgb, (x, y), (x + width, y + height), (0, 0, 255), 5)

plt.subplot(1, 1, 1)
plt.imshow(img_rgb)
plt.show()
