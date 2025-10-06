# Visual Cone Detection (and pose estimation)

## Tentative Plans

- [ ] Train YOLO v11 on cone dataset (Roboflow does this for us)

- [ ] Steal cone images from cone dataset by cropping out the given bounding boxes

- [ ] Label many keypoints on Roboflow

- [ ] Train CNN to obtain keypoints (or maybe do some more traditional CV to obtain these)

- [ ] (RANSAC) PnP on keypoints to infer cone pose

- [ ] Testing/ROS integration

- [ ] Profit!
