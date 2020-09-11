#!/bin/python3

import numpy as np
import cv2
import time
from HSV import HSVBounds
from DriveBot import DriveBot
import matplotlib.pyplot as plot

cap = cv2.VideoCapture(0)
path = [[0, 0]]
delX = 0
prevX = 0
delY = 0
prevY = 0
velCounter = 0

hsvBounds = HSVBounds(False)


class KalmanFilter:
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


kfObj = KalmanFilter()
driveBot = DriveBot(1.7, 0.5)

predictedCoords = np.zeros((2, 1), np.float32)
path = []
count = 0

while True:
    # Capture frame-by-frame
    startTime = time.time()
    ret, frame = cap.read()

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    if hsvBounds.hsvSetter:
        hsvBounds.setHSV()

    mask = cv2.inRange(hsv, hsvBounds.lowerBound, hsvBounds.upperBound)

    if not driveBot.running:
        points = cv2.findNonZero(mask)
        pointsArr = np.array(points)
        if pointsArr.size > 400:
            avg = np.mean(pointsArr, axis=0)
            if avg.size > 0:
                x = avg[0][0]
                y = avg[0][1]
                if count <= 30:
                    count += 1
                    predictedCoords = kfObj.Estimate(x, y)
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
                    predictedCoords = kfObj.Estimate(x, y)
                    path.append([predictedCoords[0], predictedCoords[1]])
                    cv2.circle(frame, (predictedCoords[0], predictedCoords[1]), 10, [0, 255, 255], -1)
                    cv2.circle(frame, (int(x), int(y)), 10, [0, 255, 0], -1)
                    print("Pos: "+str(int(x))+" "+str(int(y)))
                    print("Vel: " + str(int(delX)) + " " + str(int(delY)))
                    cv2.putText(frame, "Pos: "+str(int(x))+" "+str(int(y)), (0, 20), cv2.FONT_HERSHEY_PLAIN, 1.2, [255, 255, 255], 1)
                    cv2.putText(frame, "Vel: " + str(int(delX)) + " " + str(int(delY)), (0, 40), cv2.FONT_HERSHEY_PLAIN, 1.2, [255, 255, 255], 1)
        elif count > 30:
            predictedCoords = kfObj.kf.predict()
            path.append([predictedCoords[0], predictedCoords[1]])
            cv2.circle(frame, (predictedCoords[0], predictedCoords[1]), 10, [0, 255, 255], -1)
    else:
        if driveBot.index < len(path):
            error = np.sqrt(((driveBot.xPos - path[driveBot.index][0]) ** 2) + ((driveBot.yPos - path[driveBot.index][1]) ** 2))
            if error > 3:
                driveBot.calculateError(path[driveBot.index][0], path[driveBot.index][1])

                #Drawing the bot
                head = np.array([driveBot.xPos + np.cos(driveBot.angle) * 40, driveBot.yPos + np.sin(driveBot.angle) * 40])
                origin = np.array([driveBot.xPos, driveBot.yPos])
                cv2.circle(frame, (int(origin[0]), int(origin[1])), 10, [255, 255, 0], -1)
                cv2.line(frame, (int(origin[0]), int(origin[1])), (int(head[0]), int(head[1])), [255, 255, 0], 2)
                cv2.putText(frame, "Bot Pos: " + str(int(driveBot.xPos)) + " " + str(int(driveBot.yPos)), (0, 20), cv2.FONT_HERSHEY_PLAIN, 1.2, [255, 255, 255], 1)
                cv2.putText(frame, "Point Pos: " + str(int(path[driveBot.index][0])) + " " + str(int(path[driveBot.index][1])), (0, 40), cv2.FONT_HERSHEY_PLAIN, 1.2, [255, 255, 255], 1)
            else:
                driveBot.index += 3
                driveBot.setPointChange()
        else:
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
            count = 0

    # Drawing path
    if len(path) > 2:
        for i in range(len(path) - 1):
            cv2.line(frame, (path[i][0], path[i][1]), (path[i + 1][0], path[i + 1][1]), [0, 255, 255], 3)

    # Display the resulting frame
    cv2.imshow('frame', frame)
    cv2.imshow('mask', mask)

    key = cv2.waitKey(1) or 0xff

    if key == ord('q'):
        break
    elif key == ord('p'):
        driveBot.running = True

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
