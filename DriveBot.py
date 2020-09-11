import numpy as np

from Controller import Controller


class DriveBot:
    xPos = 0
    yPos = 0
    vel = 0
    angle = 0
    velMax = 0
    angleMax = 0
    running = False
    index = 0
    # PID control for angle
    angleControl = Controller(0.62, 0.085, 0.045)
    # PID control for velocity
    velControl = Controller(0.62, 0.085, 0.045)
    # Plots
    xPlot = np.array([])
    yPlot = np.array([])
    velPlot = np.array([])

    def __init__(self, velMax, angleMax):
        # Values to limit max angular change and velocity
        self.velMax = velMax
        self.angleMax = angleMax

    def calculateError(self, pathX, pathY):
        center = np.array([self.xPos, self.yPos])
        vector = np.array([float(pathX), float(pathY)]) - center
        # Angle calculation
        if vector[0] >= 0:
            angle = np.arctan(vector[1] / vector[0])
        else:
            angle = np.arctan(vector[1] / vector[0]) + np.deg2rad(180)
        # Error in angle
        errorAngle = angle - self.angle
        angleInput = self.angleControl.controllerOutput(errorAngle)
        # Error in distance
        errorDistance = np.sqrt(((self.xPos - float(pathX)) ** 2) + ((self.yPos - float(pathY)) ** 2))
        distanceInput = self.velControl.controllerOutput(errorDistance)
        self.Plant(angleInput, distanceInput)

    def Plant(self, angleInput, distanceInput):
        # Updating angle
        plantAngleVar = 1
        delAngle = angleInput * plantAngleVar
        delAngle = min(abs(delAngle), self.angleMax) * np.sign(delAngle)
        self.angle += delAngle
        # Updating velocity
        plantVelVar = 0.05
        delVelocity = distanceInput * plantVelVar
        print(delVelocity)
        delVelocity = min(abs(delVelocity), self.velMax) * np.sign(delVelocity)
        self.vel = delVelocity
        # Updating position
        self.xPos += np.cos(self.angle) * self.vel
        self.yPos += np.sin(self.angle) * self.vel
        self.xPlot = np.append(self.xPlot, self.xPos)
        self.yPlot = np.append(self.yPlot, self.yPos)
        self.velPlot = np.append(self.velPlot, self.vel)

    def setPointChange(self):
        # Resetting values
        self.velControl.errorI = 0
        self.velControl.errorD = 0
        self.velControl.prevError = 0
