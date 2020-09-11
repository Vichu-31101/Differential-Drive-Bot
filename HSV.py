import cv2
import numpy as np


class HSVBounds:
    lowerBound = np.array([0, 155, 162])
    upperBound = np.array([179, 255, 255])
    hsvSetter = False

    def __init__(self, setter):
        self.hsvSetter = setter

    def createSetter(self):
        def nothing(x):
            pass
        # Creating HSV tracker window
        cv2.namedWindow("Tracking")
        cv2.createTrackbar("LH", "Tracking", 0, 179, nothing)
        cv2.createTrackbar("LS", "Tracking", 0, 255, nothing)
        cv2.createTrackbar("LV", "Tracking", 0, 255, nothing)
        cv2.createTrackbar("UH", "Tracking", 179, 179, nothing)
        cv2.createTrackbar("US", "Tracking", 255, 255, nothing)
        cv2.createTrackbar("UV", "Tracking", 255, 255, nothing)

    def setHSV(self):
        # Lower bound for HSV
        lH = cv2.getTrackbarPos("LH", "Tracking")
        lS = cv2.getTrackbarPos("LS", "Tracking")
        lV = cv2.getTrackbarPos("LV", "Tracking")
        # Upper bound for HSV
        uH = cv2.getTrackbarPos("UH", "Tracking")
        uS = cv2.getTrackbarPos("US", "Tracking")
        uV = cv2.getTrackbarPos("UV", "Tracking")

        self.lowerBound = np.array([lH, lS, lV])
        self.upperBound = np.array([uH, uS, uV])
        pass

    def nothing(self):
        pass
