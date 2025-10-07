from ultralytics import YOLO
import os
import cv2
import random
DATA_PATH = "../data/fsae-cones-dataset.v1i.yolov11/test/images"

model = YOLO("../models/cones.pt")

images = list(map(lambda p: os.path.join(DATA_PATH, p), os.listdir(DATA_PATH)))

for _ in range(10):
    output = model.predict(random.choice(images))[0]
    cv2.imshow("Predicted", output.plot(labels=False))
    cv2.waitKey(0)  