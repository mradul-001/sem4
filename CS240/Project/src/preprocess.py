import numpy as np
import librosa

def preprocess(audioList):
    for audio in audioList:
        audio = librosa.load(audio)
        print(audio)
    return

preprocess(["./dataset/0_george_0.wav"])