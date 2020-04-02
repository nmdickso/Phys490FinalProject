class Config:
    def __init__(self):
        self.dataPath = "examples/quantumTomography/data/"
        # default path
        self.pdfSavePath = "examples/quantumTomography/Results/"

        # trains and finds scinet MSE this many per bar in the bargraph and
        #   takes the average
        self.averaging = 3
