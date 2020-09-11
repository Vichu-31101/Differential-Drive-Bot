class Controller:
    Kp = 0
    Ki = 0
    Kd = 0
    # Integrating Error
    errorI = 0
    # Error Difference
    errorD = 0
    prevError = 0

    def __init__(self, Kp, Ki, Kd):
        # Setting controller gains
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd

    def controllerOutput(self, error):
        # P controller
        pOutput = self.Kp * error

        # I controller
        self.errorI += error
        iOutput = self.Ki * self.errorI

        # D controller
        self.errorD = error - self.prevError
        self.prevError = error
        dOutput = self.Kd * self.errorD

        # Final output
        output = pOutput + iOutput + dOutput
        return output
