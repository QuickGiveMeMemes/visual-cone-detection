from ultralytics import YOLO

model = YOLO("../models/cones.pt")

# results = model.train(resume=True)
results = model.train(data="../data/blue-yellow-only-fsae-cones-dataset.v1i.yolov11/data.yaml", epochs=100)

print(results)