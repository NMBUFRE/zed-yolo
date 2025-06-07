# # Standalone ZED + YOLOv8 Fruit Detection Script (No ROS Required)
# # Works on Windows with ZED SDK and Ultralytics

# import csv
# import os
# import time
# import numpy as np
# from ultralytics import YOLO
# import cv2
# import pyzed.sl as sl

# class ZEDFruitDetector:
#     def __init__(self):
#         # CSV output setup
#         self.csv_path = os.path.expanduser('~/Documents/fruit_detections.csv')
#         os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
#         self.csv_file = open(self.csv_path, 'w', newline='')
#         self.writer = csv.writer(self.csv_file)
#         self.writer.writerow(['fruit_type', 'x_coordinate', 'y_coordinate'])

#         # Load YOLO model
#         self.model = YOLO("best_with_large_dataset_25_epochs.pt")  # <-- Replace with your actual model

#         # Initialize ZED camera
#         self.zed = sl.Camera()
#         init_params = sl.InitParameters()
#         init_params.camera_resolution = sl.RESOLUTION.HD720
#         init_params.depth_mode = sl.DEPTH_MODE.NEURAL
#         init_params.coordinate_units = sl.UNIT.METER
#         init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP

#         if self.zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
#             print("Failed to open ZED camera")
#             exit()

#         self.runtime_params = sl.RuntimeParameters()
#         self.image = sl.Mat()
#         self.depth = sl.Mat()

#     def run_detection_loop(self):
#         print("Starting detection loop. Press Ctrl+C to stop.")
#         try:
#             while True:
#                 if self.zed.grab(self.runtime_params) == sl.ERROR_CODE.SUCCESS:
#                     self.zed.retrieve_image(self.image, sl.VIEW.LEFT)
#                     self.zed.retrieve_measure(self.depth, sl.MEASURE.DEPTH)

#                     image_np = self.image.get_data()
#                     rgb_image = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)

#                     results = self.model(rgb_image)[0]
#                     names = self.model.names

#                     for box in results.boxes:
#                         cls_id = int(box.cls[0])
#                         fruit_type = names[cls_id]
#                         conf = float(box.conf[0])
#                         if conf < 0.5:
#                             continue

#                         x_center = int(box.xywh[0][0])
#                         y_center = int(box.xywh[0][1])

#                         depth_value = self.depth.get_value(x_center, y_center)[1]  # meters
#                         if depth_value == 0.0 or np.isnan(depth_value):
#                             continue

#                         # Estimate x, y position in camera frame
#                         camera_info = self.zed.get_camera_information()
#                         intrinsics = camera_info.camera_configuration.calibration_parameters.left_cam
#                         fx = intrinsics.fx
#                         fy = intrinsics.fy
#                         cx = intrinsics.cx
#                         cy = intrinsics.cy

#                         x = (x_center - cx) * depth_value / fx
#                         y = (y_center - cy) * depth_value / fy

#                         self.writer.writerow([fruit_type, round(x, 2), round(y, 2)])
#                         self.csv_file.flush()
#                         print(f"Detected {fruit_type} at ({round(x, 2)}, {round(y, 2)})")

#                 time.sleep(1.5)  # Limit detection frequency

#         except KeyboardInterrupt:
#             print("Detection stopped by user.")
#             self.cleanup()

#     def cleanup(self):
#         self.csv_file.close()
#         self.zed.close()


# if __name__ == '__main__':
#     detector = ZEDFruitDetector()
#     detector.run_detection_loop()


















import csv
import os
import time
import numpy as np
from ultralytics import YOLO
import cv2
import pyzed.sl as sl

class ZEDFruitDetector:
    def __init__(self, robot_position):
        """
        robot_position: tuple (x_robot, y_robot, theta)
        x_robot, y_robot: position of robot in meters on a 10x10 grid
        theta: orientation in radians (0 means facing along +X axis)
        """
        self.robot_position = robot_position  # (x, y, theta)

        # CSV output setup
        self.csv_path = os.path.expanduser('~/Documents/fruit_detections.csv')
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
        self.csv_file = open(self.csv_path, 'w', newline='')
        self.writer = csv.writer(self.csv_file)
        self.writer.writerow(['fruit_type', 'cam_x', 'cam_y', 'global_x', 'global_y'])

        # Load YOLO model
        self.model = YOLO("best_with_large_dataset_25_epochs.pt")

        # Initialize ZED camera
        self.zed = sl.Camera()
        init_params = sl.InitParameters()
        init_params.camera_resolution = sl.RESOLUTION.HD720
        init_params.depth_mode = sl.DEPTH_MODE.NEURAL
        init_params.coordinate_units = sl.UNIT.METER
        init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP

        if self.zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
            print("Failed to open ZED camera")
            exit()

        self.runtime_params = sl.RuntimeParameters()
        self.image = sl.Mat()
        self.depth = sl.Mat()

    def run_detection_loop(self):
        print("Starting detection loop. Press Ctrl+C to stop.")
        try:
            while True:
                if self.zed.grab(self.runtime_params) == sl.ERROR_CODE.SUCCESS:
                    self.zed.retrieve_image(self.image, sl.VIEW.LEFT)
                    self.zed.retrieve_measure(self.depth, sl.MEASURE.DEPTH)

                    image_np = self.image.get_data()
                    rgb_image = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)

                    results = self.model(rgb_image)[0]
                    names = self.model.names

                    for box in results.boxes:
                        cls_id = int(box.cls[0])
                        fruit_type = names[cls_id]
                        conf = float(box.conf[0])
                        if conf < 0.5:
                            continue

                        x_center = int(box.xywh[0][0])
                        y_center = int(box.xywh[0][1])

                        depth_value = self.depth.get_value(x_center, y_center)[1]
                        if depth_value == 0.0 or np.isnan(depth_value):
                            continue

                        # Get camera intrinsics
                        camera_info = self.zed.get_camera_information()
                        intrinsics = camera_info.camera_configuration.calibration_parameters.left_cam
                        fx = intrinsics.fx
                        fy = intrinsics.fy
                        cx = intrinsics.cx
                        cy = intrinsics.cy

                        # Position in camera frame (cam_x, cam_y)
                        cam_x = (x_center - cx) * depth_value / fx
                        cam_y = (y_center - cy) * depth_value / fy

                        # Global position in robot frame
                        x_r, y_r, theta = self.robot_position
                        global_x = x_r + cam_x * np.cos(theta) - cam_y * np.sin(theta)
                        global_y = y_r + cam_x * np.sin(theta) + cam_y * np.cos(theta)

                        self.writer.writerow([fruit_type, round(cam_x, 2), round(cam_y, 2), round(global_x, 2), round(global_y, 2)])
                        self.csv_file.flush()
                        print(f"Detected {fruit_type}: cam=({round(cam_x,2)}, {round(cam_y,2)}), global=({round(global_x,2)}, {round(global_y,2)})")

                time.sleep(1.5)

        except KeyboardInterrupt:
            print("Detection stopped by user.")
            self.cleanup()

    def cleanup(self):
        self.csv_file.close()
        self.zed.close()

if __name__ == '__main__':
    # Simulated robot position on 10x10 grid: (x, y, theta in radians)
    robot_position = (1.0, 1.0, 0.0)  # Starts at (1,1) facing +X direction
    detector = ZEDFruitDetector(robot_position)
    detector.run_detection_loop()
