import cv2
import yaml
import os
import matplotlib.pyplot as plt


AUGMENT_ANGLES = (-10, 10, 0)  # degrees
MIN_ALLOWED_CONE_HEIGHT = 60  # pixels

MIN_ASPECT_RATIO = 0.5
MAX_ASPECT_RATIO = 0.9

DATA_PATH = "../data/fsae-cones-dataset.v1i.yolov11"
OUT_PATH = "../data/cones"
DATASET = "train"
NUM_FILES_IN_DATASET = len(os.listdir(os.path.join(DATA_PATH, DATASET, "images")))
FILE_HEADERS_IN_DATASET = list(
    map(lambda f: f[:-4], os.listdir(os.path.join(DATA_PATH, DATASET, "images")))
)

GOOD_FILE_HEADERS_IN_DATASET = set(map(lambda f: f[:-4], os.listdir("../data/good")))


def generate_subimages(image, center, size, i, j):
    # Augment rotations
    for angle in AUGMENT_ANGLES:
        M = cv2.getRotationMatrix2D(center, angle, 1)

        img_rot = cv2.warpAffine(image, M, image.shape[:2][::-1])

        cropped = cv2.getRectSubPix(img_rot, size, center)

        o_fname = os.path.join(
            OUT_PATH, f"{FILE_HEADERS_IN_DATASET[i]}_{j}_{angle}.jpg"
        )
        cv2.imwrite(o_fname, cropped)
        print(f"Written cone image {o_fname}")


if not os.path.exists(OUT_PATH):
    os.mkdir(OUT_PATH)
print(f"Created output directory {OUT_PATH}.")

with open(os.path.join(DATA_PATH, "data.yaml"), "r") as f:
    data = yaml.safe_load(f)
print(
    f"Successfully read in data configuration from {os.path.join(DATA_PATH, "data.yml")}."
)

labels = data["names"]

good = os.listdir()
widths = []
heights = []
aspect_ratios = []

for i in range(NUM_FILES_IN_DATASET):
    if FILE_HEADERS_IN_DATASET[i] not in GOOD_FILE_HEADERS_IN_DATASET:
        continue

    image = cv2.imread(
        os.path.join(DATA_PATH, DATASET, "images", FILE_HEADERS_IN_DATASET[i] + ".jpg"),
        cv2.IMREAD_COLOR,
    )

    with open(
        os.path.join(DATA_PATH, DATASET, "labels", FILE_HEADERS_IN_DATASET[i] + ".txt"),
        "r",
    ) as boxes:

        for j, box in enumerate(boxes):
            id, center_x, center_y, width, height = map(float, box.split())

            id = labels[int(id)]

            if id not in ("blue_cone", "yellow_cone"):
                continue

            full_height, full_width = image.shape[:2]

            center = (int(center_x * full_width), int(center_y * full_height))
            size = (int(width * full_width), int(height * full_height))

            widths.append(size[0])
            heights.append(size[1])
            aspect_ratios.append(size[0] / size[1])

            if (
                size[1] < MIN_ALLOWED_CONE_HEIGHT
                or not MIN_ASPECT_RATIO <= size[0] / size[1] <= MAX_ASPECT_RATIO
            ):
                continue

            generate_subimages(image, center, size, i, j)


fig, ax = plt.subplots(ncols=3)
ax[0].set_title("aspect ratios")
ax[0].hist(widths, bins=100)

ax[1].set_title("heights")
ax[1].hist(heights, bins=100)

ax[2].set_title("aspect ratios")
ax[2].hist(aspect_ratios, bins=100)

plt.show()
