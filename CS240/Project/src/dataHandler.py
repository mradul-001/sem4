from processing import Processor
import yfinance as yf
import pandas as pd
import json

class DataLoaderAndSaver:

    def __init__(self):
        self.TICKERFILE         = "./src/tickers.json"
        self.RAWDIRECTORY       = "./data/raw"
        self.PROCESSEDDIRECTORY = "./data/processed"
        self.PERIOD             = "5y"
        return

    def loadList(self):
        try:
            with open(self.TICKERFILE, "r") as file:
                tickerList = json.load(file)
                return tickerList
        except (FileNotFoundError, json.JSONDecodeError) as error:
            print("Error: ", error)
            return []

    def fetchData(self):
        tickerList  = self.loadList()
        dataframe   = dict()
        for ticker in tickerList:
            stock   = yf.Ticker(ticker=ticker)
            df      = stock.history(period=self.PERIOD)
            dataframe[ticker] = df
        return dataframe

    def fetchAndSaveData(self):
        dataFrameList = self.fetchData()
        for name, data in dataFrameList.items():
            try:
                with open(f"{self.RAWDIRECTORY}/{name[:-3]}.csv", "w") as file:
                    data.to_csv(file, index=True)
            except FileNotFoundError:
                print("Error in saving the data.")
                return
        return

    def processAndSaveData(self):
        tickerList = self.loadList()
        for ticker in tickerList:
            dataframe = pd.read_csv(f"{self.RAWDIRECTORY}/{ticker[:-3]}.csv")
            processor = Processor(ticker[:-3], self.PROCESSEDDIRECTORY, dataframe)
            processor.saveData()
        return
    
    def getProcessedData(self, tickerSymbol):
        """
        Return the processed data for the given stock as a dataframe
        """
        data = pd.read_csv(f"{self.PROCESSEDDIRECTORY}/{tickerSymbol}.csv")
        return data