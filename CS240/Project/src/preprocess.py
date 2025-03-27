import numpy as np
import librosa
import glob
import matplotlib.pyplot as plt

class Dataset:
    
    # Why 16kHz?:  https://cloud.google.com/speech-to-text/docs/optimizing-audio-files-for-speech-to-text#sample_rate_frequency_range
    
    def __init__(self, rootDir, samplingRate = 16000, nmfcc = 20):
        self.rootDir        = rootDir
        self.samplingRate   = samplingRate
        self.filePaths      = []
        self.audios         = []
        self.trimmedAudios  = []
        self.mfccs          = []
        self.nmfcc          = nmfcc
        return

    def listPaths(self):
        fileRegex = f"{self.rootDir}/dataset/*.wav"
        self.filePaths = glob.glob(fileRegex)
        return
    
    def loadAudio(self):
        self.audios = [librosa.load(path, sr=self.samplingRate)[0] for path in self.filePaths]
        return
        
    def trimAndPad(self):
        lengths = np.array([audio.shape[0] for audio in self.audios])
        cap     = int(np.percentile(lengths, 90))
        for audio in self.audios:
            if audio.shape[0] >= cap:
                self.trimmedAudios.append(audio[:cap])
            else:
                paddedAudio = np.pad(audio, (0, cap - audio.shape[0]))
                self.trimmedAudios.append(paddedAudio)
        self.trimmedAudios = [audio / np.max(np.abs(audio)) for audio in self.trimmedAudios]
        return

    def featureExtraction(self):
        self.mfccs = [librosa.feature.mfcc(y = audio, sr = self.samplingRate, n_mfcc = self.nmfcc) for audio in self.trimmedAudios]
        return

    def printInfo(self):
        print("-" * 40)
        print("|" + " RESULTS OF AUDIO PREPROCESSING ".center(38) + "|")
        print("-" * 40)
        print("-" * 40)
        print("Trimmed Audio: ".center(40))
        print("-" * 40)
        print("Number of audios".ljust(28) + ":", len(self.trimmedAudios))
        print("Length of each audio".ljust(28) + ":", self.trimmedAudios[0].shape[0])
        print()
        print("-" * 40)
        print("Features Extracted:".center(40))
        print("-" * 40)
        print("Type".ljust(28) + ":", "MFCC")
        print("MFCC count for each audio".ljust(28) + ":", self.nmfcc)
        print()
        print()
        return


    def processData(self):
        self.listPaths()
        self.loadAudio()
        self.trimAndPad()
        self.featureExtraction()
        self.printInfo()
        return
    

d = Dataset("./")
d.processData()