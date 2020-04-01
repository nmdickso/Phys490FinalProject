class Config:
    def __init__(self):
        self.dataPath="quantumTomography/data/"
        #default path
        self.pdfSavePath="quantumTomography/Results/"

        #trains and finds scinet MSE this many per bar in the bargraph and takes the average
        self.averaging=3