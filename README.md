# Visual Cone Detection (and pose estimation)

## Tentative Plans

- [x] Train YOLO v11 on cone dataset (Roboflow does this for us)

- [x] Steal cone images from cone dataset by cropping out the given bounding boxes

- [x] Label many keypoints on Roboflow

- [x] Train CNN to obtain keypoints 

- [x] (RANSAC) PnP on keypoints to infer cone pose

- [ ] Testing/ROS integration

- [ ] Profit!


## File Structure
- `monocular`: keypoint calculation and PnP

- `stereo`: stereo

- `util`: miscellaneous helper scripts
