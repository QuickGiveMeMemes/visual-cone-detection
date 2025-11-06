from calibration import Calibrator
import os

calibrator = Calibrator(
    0.1,
    (8, 5),
    "basler",
    "testing_cam_cal_mgs_old",
    (1920, 1200),
)
files = [
    "testing_cam_cal_mgs_old/" + f
    for f in os.listdir("testing_cam_cal_mgs_old")
]

calibrator.load_saved_images(files)
calibrator.generate_calibration(
    "calDataOld"
)