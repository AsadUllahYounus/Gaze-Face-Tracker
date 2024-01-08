# Import necessary libraries
import cv2 as cv
import numpy as np
import mediapipe as mp
import socket
import argparse
import time
import csv
from datetime import datetime
import os
import threading
import tkinter as tk
from tkinter import ttk

# Define the EyeTrackingApp class
class EyeTrackingApp:
    # Define constants for facial landmarks
    LEFT_IRIS = [474, 475, 476, 477]
    RIGHT_IRIS = [469, 470, 471, 472]
    L_H_LEFT = [33]  # Left eye Left Corner
    L_H_RIGHT = [133]  # Left eye Right Corner
    R_H_LEFT = [362]  # Right eye Left Corner
    R_H_RIGHT = [263]  # Right eye Right Corner
    RIGHT_EYE_POINTS = [33, 160, 159, 158, 133, 153, 145, 144]
    LEFT_EYE_POINTS = [362, 385, 386, 387, 263, 373, 374, 380]

    # Initialize EyeTrackingApp instance
    def __init__(self):
        self.initialize_parameters()
        self.initialize_components()

    # Initialize parameters for eye tracking
    def initialize_parameters(self):
        self.PRINT_DATA = True
        self.DEFAULT_WEBCAM = 0
        self.SHOW_ALL_FEATURES = True
        self.LOG_DATA = True
        self.LOG_ALL_FEATURES = False
        self.LOG_FOLDER = "logs"
        self.SERVER_IP = "127.0.0.1"
        self.SERVER_PORT = 7070
        self.SHOW_BLINK_COUNT_ON_SCREEN = True
        self.TOTAL_BLINKS = 0
        self.EYES_BLINK_FRAME_COUNTER = 0
        self.BLINK_THRESHOLD = 0.51
        self.EYE_AR_CONSEC_FRAMES = 2
        self.is_detecting = False  # Flag for detection status

    # Initialize components such as face mesh and camera
    def initialize_components(self):
        if self.PRINT_DATA:
            print("Initializing the face mesh and camera...")
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.cam_source = int(self.DEFAULT_WEBCAM)
        self.cap = cv.VideoCapture(self.cam_source)
        self.iris_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.csv_data = []
        if not os.path.exists(self.LOG_FOLDER):
            os.makedirs(self.LOG_FOLDER)
        self.column_names = [
            "Timestamp (ms)",
            "Left Eye Center X",
            "Left Eye Center Y",
            "Right Eye Center X",
            "Right Eye Center Y",
            "Left Iris Relative Pos Dx",
            "Left Iris Relative Pos Dy",
            "Right Iris Relative Pos Dx",
            "Right Iris Relative Pos Dy",
            "Total Blink Count"
        ]
        if self.LOG_ALL_FEATURES:
            self.column_names.extend(
                [f"Landmark_{i}_X" for i in range(468)]
                + [f"Landmark_{i}_Y" for i in range(468)]
            )

    # Calculate vector position between two points
    def calculate_vector_position(self, point1, point2):
        x1, y1 = point1.ravel()
        x2, y2 = point2.ravel()
        return x2 - x1, y2 - y1

    # Calculate 3D Euclidean distance between points
    def euclidean_distance_3D(self, points):
        P0, P3, P4, P5, P8, P11, P12, P13 = points
        numerator = (
            np.linalg.norm(P3 - P13) ** 3
            + np.linalg.norm(P4 - P12) ** 3
            + np.linalg.norm(P5 - P11) ** 3
        )
        denominator = 3 * np.linalg.norm(P0 - P8) ** 3
        distance = numerator / denominator
        return distance

    # Calculate blinking ratio based on landmarks
    def blinking_ratio(self, landmarks):
        right_eye_ratio = self.euclidean_distance_3D(landmarks[self.RIGHT_EYE_POINTS])
        left_eye_ratio = self.euclidean_distance_3D(landmarks[self.LEFT_EYE_POINTS])
        ratio = (right_eye_ratio + left_eye_ratio + 1) / 2
        return ratio

    # Process each frame for eye tracking
    def process_frame(self, frame):
        ret, frame = self.cap.read()
        if not ret:
            return
        frame = cv.flip(frame, 1)
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        img_h, img_w = frame.shape[:2]
        results = self.mp_face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            mesh_points = np.array(
                [
                    np.multiply([p.x, p.y], [img_w, img_h]).astype(int)
                    for p in results.multi_face_landmarks[0].landmark
                ]
            )

            mesh_points_3D = np.array(
                [[n.x, n.y, n.z] for n in results.multi_face_landmarks[0].landmark]
            )

            eyes_aspect_ratio = self.blinking_ratio(mesh_points_3D)

            if eyes_aspect_ratio <= self.BLINK_THRESHOLD:
                self.EYES_BLINK_FRAME_COUNTER += 1
            else:
                if self.EYES_BLINK_FRAME_COUNTER > self.EYE_AR_CONSEC_FRAMES:
                    self.TOTAL_BLINKS += 1
                self.EYES_BLINK_FRAME_COUNTER = 0

            if self.SHOW_BLINK_COUNT_ON_SCREEN:
                cv.putText(
                    frame,
                    f"Blinks: {self.TOTAL_BLINKS}",
                    (30, 50),
                    cv.FONT_HERSHEY_DUPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                    cv.LINE_AA,
                )

            if self.SHOW_ALL_FEATURES:
                for point in mesh_points:
                    cv.circle(frame, tuple(point), 1, (0, 255, 0), -1)

            (l_cx, l_cy), l_radius = cv.minEnclosingCircle(mesh_points[self.LEFT_IRIS])
            (r_cx, r_cy), r_radius = cv.minEnclosingCircle(mesh_points[self.RIGHT_IRIS])
            center_left = np.array([l_cx, l_cy], dtype=np.int32)
            center_right = np.array([r_cx, r_cy], dtype=np.int32)

            cv.circle(frame, center_left, int(l_radius), (255, 0, 255), 2, cv.LINE_AA)
            cv.circle(frame, center_right, int(r_radius), (255, 0, 255), 2, cv.LINE_AA)
            cv.circle(frame, mesh_points[self.L_H_RIGHT][0], 3, (255, 255, 255), -1, cv.LINE_AA)
            cv.circle(frame, mesh_points[self.L_H_LEFT][0], 3, (0, 255, 255), -1, cv.LINE_AA)
            cv.circle(frame, mesh_points[self.R_H_RIGHT][0], 3, (255, 255, 255), -1, cv.LINE_AA)
            cv.circle(frame, mesh_points[self.R_H_LEFT][0], 3, (0, 255, 255), -1, cv.LINE_AA)

            l_dx, l_dy = self.calculate_vector_position(mesh_points[self.L_H_LEFT], center_left)
            r_dx, r_dy = self.calculate_vector_position(mesh_points[self.R_H_LEFT], center_right)

            if self.PRINT_DATA:
                print(f"Total Blinks: {self.TOTAL_BLINKS}")
                print(f"Left Eye Center X: {l_cx} Y: {l_cy}")
                print(f"Right Eye Center X: {r_cx} Y: {r_cy}")
                print(f"Left Iris Relative Pos Dx: {l_dx} Dy: {l_dy}")
                print(f"Right Iris Relative Pos Dx: {r_dx} Dy: {r_dy}\n")

            if self.LOG_DATA:
                timestamp = int(time.time() * 1000)
                log_entry = [timestamp, l_cx, l_cy, r_cx, r_cy, l_dx, l_dy, r_dx, r_dy, self.TOTAL_BLINKS]
                self.csv_data.append(log_entry)
                if self.LOG_ALL_FEATURES:
                    log_entry.extend([p for point in mesh_points for p in point])
                self.csv_data.append(log_entry)

            packet = np.array([l_cx, l_cy, l_dx, l_dy], dtype=np.int32)
            self.iris_socket.sendto(bytes(packet), (self.SERVER_IP, self.SERVER_PORT))

        cv.imshow("Eye Tracking", frame)

    # Run the eye tracking detection
    def run(self):
        try:
            while self.is_detecting:
                ret, frame = self.cap.read()
                if not ret:
                    break
                self.process_frame(frame)
                if cv.waitKey(1) & 0xFF == ord("q"):
                    print("Exiting program...")
                    break
        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            self.release_resources()

    # Start the eye tracking detection in a separate thread
    def start_detection(self):
        self.is_detecting = True
        threading.Thread(target=self.run).start()

    # Stop the eye tracking detection
    def stop_detection(self):
        self.is_detecting = False

    # Release resources such as camera and sockets
    def release_resources(self):
        self.cap.release()
        cv.destroyAllWindows()
        self.iris_socket.close()
        print("Program exited successfully.")
        self.write_to_csv()

    # Write collected data to CSV file
    def write_to_csv(self):
        if self.LOG_DATA:
            print("Writing data to CSV...")
            timestamp_str = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
            csv_file_name = os.path.join(
                self.LOG_FOLDER, f"eye_tracking_log_{timestamp_str}.csv"
            )
            with open(csv_file_name, "w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(self.column_names)
                writer.writerows(self.csv_data)
            print(f"Data written to {csv_file_name}")

# Define EyeTrackingGUI class
class EyeTrackingGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Eye Tracking GUI")
        self.eye_tracking_app = EyeTrackingApp()
        self.start_button = ttk.Button(
            master, text="Start Detecting", command=self.start_detection
        )
        self.start_button.pack(pady=10)
        self.stop_button = ttk.Button(
            master, text="Stop Detecting", command=self.stop_detection
        )
        self.stop_button.pack(pady=10)
        self.quit_button = ttk.Button(master, text="Quit", command=self.quit_app)
        self.quit_button.pack(pady=10)

    # Start the eye tracking detection
    def start_detection(self):
        if not self.eye_tracking_app.is_detecting:
            self.eye_tracking_app.start_detection()

    # Stop the eye tracking detection
    def stop_detection(self):
        if self.eye_tracking_app.is_detecting:
            self.eye_tracking_app.stop_detection()

    # Quit the application and release resources
    def quit_app(self):
        self.eye_tracking_app.release_resources()
        self.master.destroy()

# Create the main Tkinter window
root = tk.Tk()
# Create an instance of EyeTrackingGUI
eye_tracking_gui = EyeTrackingGUI(root)
# Start the Tkinter event loop
root.mainloop()
