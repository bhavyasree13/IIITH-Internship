import cv2
from ultralytics import YOLO

img_pth = r"https://www.alltheedge.com/wp-content/uploads/2018/02/Rain.jpg"
model = YOLO("yolov8n.pt")

results = model(source=img_pth)
res_plotted = results[0].plot()

cv2.imshow("result", res_plotted)
cv2.waitKey(0)
