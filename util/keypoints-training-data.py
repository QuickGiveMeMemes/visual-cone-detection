import cv2
import yaml
import os

DATA_PATH = "../data/fsae-cones-dataset.v1i.yolov11"
OUT_PATH = "../data/cones"
DATASET = "train"
NUM_FILES_IN_DATASET = len(os.listdir(os.path.join(DATA_PATH, "test")))
FILE_HEADERS_IN_DATASET = list(map(lambda f: f[:-4], os.listdir(os.path.join(DATA_PATH, "test"))))

if not os.path.exists(OUT_PATH):
   os.mkdir(OUT_PATH)
print(f"Created output directory {OUT_PATH}.")

with open(os.path.join(DATA_PATH, "data.yml"), "r") as f:
   data = yaml.safe_load(f)
print(f"Successfully read in data configuration from {os.path.join(DATA_PATH, "data.yml")}.")

labels = data["names"]

for i in range(NUM_FILES_IN_DATASET):
   image = cv2.imread(os.path.join(DATA_PATH, DATASET, "images", FILE_HEADERS_IN_DATASET[i]), cv2.IMREAD_COLOR)

   with open(os.path.join(DATA_PATH, DATASET, "labels", FILE_HEADERS_IN_DATASET[i]), "r") as boxes:

      for j, box in enumerate(boxes):
         id, center_x, center_y, width, height = box.split()
         id = labels[id]

         if id not in ("blue_cone", "yellow_cone"): 
            continue

         full_height, full_width = image.shape[:2]
         
         y_lb = int((center_y - height / 2.0) * full_height)
         y_ub = int((center_y + height / 2.0) * full_height)
         x_lb = int((center_x - width / 2.0) * full_width)
         x_ub = int((center_x + width / 2.0) * full_width)
         
         cropped = image[y_lb : y_ub, x_lb : x_ub]
         
         o_fname = os.path.join(OUT_PATH, f"{FILE_HEADERS_IN_DATASET[i]}_{j}.jpg")
         cv2.imwrite(o_fname, cropped)

         print(f"Written cone image {o_fname} with bounds: (x_lb: {x_lb}, x_ub: {x_ub}, y_lb: {y_lb}, y_ub: {y_ub})")

