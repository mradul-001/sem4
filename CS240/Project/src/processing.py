import pandas as pd
import numpy as np

class Processor:

    def __init__(self, stock, saveDirectory, dataframe):
        self.stock          = stock
        self.saveDirectory  = saveDirectory
        self.dataframe      = dataframe.drop({"High", "Low", "Dividends", "Stock Splits"}, axis = 1)
        return
    
    # [Date, Open, Close, Volume]

    def removeUnfilled(self):
        for column in self.dataframe.columns:
            if column == "Date":
                continue
            mean = self.dataframe[column].mean()
            self.dataframe[column] = self.dataframe[column].fillna(mean)
        return

    def standardize(self):
        for column in self.dataframe.columns:
            if column == "Date":
                continue
            mean = self.dataframe[column].mean()
            std  = self.dataframe[column].std()
            if std != 0:
                self.dataframe[column] = (self.dataframe[column] - mean) / std
            else:
                self.dataframe[column] = 0
        return
    
    def saveData(self):
        """
        Process and save the processed data in the given directory
        """
        self.removeUnfilled()
        self.standardize()
        self.dataframe.set_index("Date", inplace=True)
        try:
            with open(f"{self.saveDirectory}/{self.stock}.csv", "w") as file:
                self.dataframe.to_csv(file)
        except(FileNotFoundError):
            print("Error saving processed data.")
            return
        return