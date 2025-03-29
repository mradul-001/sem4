from dataHandler import DataLoaderAndSaver

dataloader = DataLoaderAndSaver()

# fetching "1d" data for "5y"
dataloader.fetchAndSaveData("1d", "5y")
dataloader.processAndSaveData("1d")

# fetching "5m" data for "6m"
dataloader.fetchAndSaveData("5m", "1mo")
dataloader.processAndSaveData("5m")