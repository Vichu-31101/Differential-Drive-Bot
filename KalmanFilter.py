import cv2
import numpy as np


class KalmanFilter:
    # Output as postion and velocity with input as position
    kf = cv2.KalmanFilter(4, 2)
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.05  # process uncertainty
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 10  # measurement noise uncertainty

    def Estimate(self, coordX, coordY):
        measured = np.array([[np.float32(coordX)], [np.float32(coordY)]])
        self.kf.correct(measured)
        predicted = self.kf.predict()
        return predicted

    def Predict(self):
        predicted = self.kf.predict()
        return predicted
