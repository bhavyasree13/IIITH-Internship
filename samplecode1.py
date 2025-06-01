import cv2
from ultralytics import YOLO
img_pth = "C:/Users/PARTHAV/Downloads/bus.jpg"
model = YOLO("yolov8n.pt") 
results = model(source=img_pth)
res_plotted = results[0].plot()
cv2.imshow("result", res_plotted)
cv2.waitKey(0)


