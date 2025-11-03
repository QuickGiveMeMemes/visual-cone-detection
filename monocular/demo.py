import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
import numpy as np
from ultralytics import YOLO
from keypoint_detection import KeypointDetector
from torchvision import transforms
import torch
from pose_estimation import ConeEstimator

DEVICE = "cpu"
LOAD_FILE = "../models/test_run_cont8_epoch_300.pth"

cam = cv2.VideoCapture(0)
yolo = YOLO("../models/cones.pt")

estimator = ConeEstimator(
    K=np.array([
        [638.3876756065123, 0.0, 338.8268432747065],
        [0.0, 638.2673694583794, 244.95855453271443],
        [0.0, 0.0, 1.0],
    ]),
    D=np.array([
        [
            0.0026993034316860564,
            0.02835225042333684,
            0.0022198978341294575,
            0.0009381250289687192,
            -0.1907897902662239,
        ]
    ]),
)

keypoint_model = KeypointDetector().to(DEVICE)
keypoint_model.load_state_dict(torch.load(LOAD_FILE, map_location=DEVICE))
keypoint_model.eval()

plt.ion() 
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.view_init(elev=-25, azim=-30, roll=-82)

def plot_cone(ax, height=0.3, radius=0.1, resolution=20, R=None, t=None, color='orange'):
    theta = np.linspace(0, 2*np.pi, resolution)
    z = np.linspace(0, height, 2)
    T, Z = np.meshgrid(theta, z)
    Rr = radius * (1 - Z/height)
    X = Rr * np.cos(T)
    Y = Rr * np.sin(T)

    points = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)
    if R is not None and t is not None:
        points = (R @ points.T).T + t.flatten()
    X = points[:,0].reshape(Z.shape)
    Y = points[:,1].reshape(Z.shape)
    Z = points[:,2].reshape(Z.shape)
    ax.plot_surface(X, Y, Z, color=color, alpha=0.8)


while True:
    ret, img = cam.read()
    if not ret:
        break

    output = yolo.predict(img)[0]

    cropped, sizes, corner = [], [], []

    for result in output:
        p1 = np.asarray(result.boxes.xyxyn[0][:2] * np.array(img.shape[:2][::-1]), dtype=int)
        p2 = np.asarray(result.boxes.xyxyn[0][2:] * np.array(img.shape[:2][::-1]), dtype=int)

        cv2.rectangle(img, p1, p2, (255,0,0), 2)
        box = img[p1[1]:p2[1], p1[0]:p2[0]]
        box = cv2.resize(box, (80,80))
        cropped.append(transforms.ToTensor()(box))
        sizes.append(p2 - p1)
        corner.append(p1)

    ax.cla()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_box_aspect([1, 1, 3 / 2])

    ax.set_title("Poses")

    ax.set_xlim(-1, 1) 
    ax.set_ylim(-1, 1) 
    ax.set_zlim(0, 3)  

    ax.quiver(0, 0, 0, 1, 0, 0, length=1, color='r', linewidth=2)
    ax.quiver(0, 0, 0, 0, 1, 0, length=1, color='g', linewidth=2) 
    ax.quiver(0, 0, 0, 0, 0, 1, length=1, color='b', linewidth=2) 


    if len(cropped) > 0:
        out = keypoint_model(torch.from_numpy(np.array(cropped, dtype=np.float32))).detach()

        for i, cone in enumerate(out):
            x = cone[::2] * sizes[i][0] + corner[i][0]
            y = cone[1::2] * sizes[i][1] + corner[i][1]

            cv2.circle(img, (int(corner[i][0]), int(corner[i][1])), 3, (255, 0, 0), -1)

            for xx, yy in zip(x, y):
                cv2.circle(img, (int(xx), int(yy)), 3, (0, 255, 0), -1)

            keypoints_2d = np.column_stack((x, y))
            success, rvec, tvec = estimator.estimate_pose(keypoints_2d)

            print(tvec)

            if success:
                color = np.random.rand(3)
                R, _ = cv2.Rodrigues(rvec)
                plot_cone(ax, height=0.3, radius=0.1, R=R, t=tvec, color=color)


    cv2.imshow("Keypoints", img)
    plt.pause(0.001)
    cv2.waitKey(1)

