#!/bin/python3

import numpy as np
import cv2
import matplotlib.pyplot as plot

from HSV import HSVBounds
from DriveBot import DriveBot
from KalmanFilter import KalmanFilter
# Variables
cap = cv2.VideoCapture(0)
delX = 0
prevX = 0
delY = 0
prevY = 0
velCounter = 0
initCounter = 0
predictedCords = np.zeros((2, 1), np.float32)
path = []


# Class objects
# Set value as True for HSV setter
hsvBounds = HSVBounds(False)
kalmanFilter = KalmanFilter()
# Setting velMax and angleMax
driveBot = DriveBot(1.7, 0.5)


while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Converting to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    if hsvBounds.hsvSetter:
        hsvBounds.setHSV()

    # Determining the mask using HSV bounds
    mask = cv2.inRange(hsv, hsvBounds.lowerBound, hsvBounds.upperBound)

    # To check if the bot is running
    if not driveBot.running:
        # Obtaining the position of white pixels from the mask
        points = cv2.findNonZero(mask)
        pointsArr = np.array(points)
        # To check if object is available
        if pointsArr.size > 400:
            avg = np.mean(pointsArr, axis=0)
            if avg.size > 0:
                x = avg[0][0]
                y = avg[0][1]
                # To close the kalman filter in on true value
                if initCounter <= 30:
                    initCounter += 1
                    predictedCords = kalmanFilter.Estimate(x, y)
                    prevX = x
                    prevY = y
                else:
                    if velCounter <= 10:
                        velCounter += 1
                    else:
                        delX = x - prevX
                        delY = y - prevY
                        prevX = x
                        prevY = y
                    # Using kalman filter to measure, correct and predict current position
                    predictedCords = kalmanFilter.Estimate(x, y)
                    # Adding points to the path
                    path.append([predictedCords[0], predictedCords[1]])
                    # Drawing elements and displaying
                    cv2.circle(frame, (predictedCords[0], predictedCords[1]), 10, [0, 255, 255], -1)
                    cv2.circle(frame, (int(x), int(y)), 10, [0, 255, 0], -1)
                    print("Pos: " + str(int(x)) + " " + str(int(y)))
                    print("Vel: " + str(int(delX)) + " " + str(int(delY)))
                    cv2.putText(frame, "Pos: " + str(int(x)) + " " + str(int(y)), (0, 20), cv2.FONT_HERSHEY_PLAIN, 1.2,
                                [255, 255, 255], 1)
                    cv2.putText(frame, "Vel: " + str(int(delX)) + " " + str(int(delY)), (0, 40), cv2.FONT_HERSHEY_PLAIN,
                                1.2, [255, 255, 255], 1)
        elif initCounter > 30:
            # Occlusion
            # Using kalman filter to predict position using saved velocity
            predictedCords = kalmanFilter.Predict()
            path.append([predictedCords[0], predictedCords[1]])
            cv2.circle(frame, (predictedCords[0], predictedCords[1]), 10, [0, 255, 255], -1)
    else:
        # Drive bot running
        if driveBot.index < len(path):
            error = np.sqrt(((driveBot.xPos - path[driveBot.index][0]) ** 2) + ((driveBot.yPos - path[driveBot.index][1]) ** 2))
            # To check is the bot is close to the set point
            if error > 2:
                driveBot.calculateError(path[driveBot.index][0], path[driveBot.index][1])

                # Drawing the bot
                head = np.array(
                    [driveBot.xPos + np.cos(driveBot.angle) * 40, driveBot.yPos + np.sin(driveBot.angle) * 40])
                origin = np.array([driveBot.xPos, driveBot.yPos])
                cv2.circle(frame, (int(origin[0]), int(origin[1])), 10, [255, 255, 0], -1)
                cv2.line(frame, (int(origin[0]), int(origin[1])), (int(head[0]), int(head[1])), [255, 255, 0], 2)
                cv2.putText(frame, "Bot Pos: " + str(int(driveBot.xPos)) + " " + str(int(driveBot.yPos)), (0, 20),
                            cv2.FONT_HERSHEY_PLAIN, 1.2, [255, 255, 255], 1)
                cv2.putText(frame,
                            "Point Pos: " + str(int(path[driveBot.index][0])) + " " + str(int(path[driveBot.index][1])),
                            (0, 40), cv2.FONT_HERSHEY_PLAIN, 1.2, [255, 255, 255], 1)
            else:
                # Else change the set point to next point
                driveBot.index += 3
                driveBot.setPointChange()
        else:
            # Bot completed travel
            path.clear()
            timeSpace = np.linspace(0, 100, driveBot.xPlot.size)
            fig, axs = plot.subplots(2, 2)
            axs[0][0].set_title('X Pos')
            axs[0][0].plot(timeSpace, driveBot.xPlot)
            axs[0][1].set_title('Y Pos')
            axs[0][1].plot(timeSpace, driveBot.yPlot)
            axs[1][0].set_title('Vel')
            axs[1][0].plot(timeSpace, driveBot.velPlot)
            axs[1][1].set_title('X-Y')
            axs[1][1].plot(driveBot.xPlot, driveBot.yPlot)
            plot.show()
            driveBot.running = False
            initCounter = 0

    # Drawing path
    if len(path) > 2:
        for i in range(len(path) - 1):
            cv2.line(frame, (path[i][0], path[i][1]), (path[i + 1][0], path[i + 1][1]), [0, 255, 255], 3)

    # Display the resulting frame
    cv2.imshow('frame', frame)
    cv2.imshow('mask', mask)

    key = cv2.waitKey(1) or 0xff

    # q key to quit and p key to start bot
    if key == ord('q'):
        break
    elif key == ord('p'):
        driveBot.running = True

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
