# -*- coding: utf-8 -*-
"""
@author: diyasinha
"""
import cv2, os
from model import EmotionDetction
import numpy as np

class EmoPlayer:
    def __init__(self):
        self.model = EmotionDetction("face_model.json", "face_model.h5")
    
    def rec_emo(self,fc):
        roi = cv2.resize(fc, (48, 48))
        pred = self.model.predict_emotion(roi[np.newaxis, :, :, np.newaxis])
        return pred
    
    def play_songs(self, emo):
        if emo == "Happy":
            print("Happy Song")
            os.system(r'C:\Users\cutyp\.spyder-py3\FinalYearProject\songs\mercy.mp3')
        elif emo == "Angry":
            print("Angry Song")
            os.system(r'C:\Users\cutyp\.spyder-py3\FinalYearProject\songs\mercy.mp3')
        elif emo == "Sad":
            print("Sad Song")
            os.system(r'C:\Users\cutyp\.spyder-py3\FinalYearProject\songs\mercy.mp3')
        elif emo == "Surprise":
            print("Surprise Song")
            os.system(r'C:\Users\cutyp\.spyder-py3\FinalYearProject\songs\mercy.mp3')
        else:
            print("Neutral Song")
            os.system(r'C:\Users\cutyp\.spyder-py3\FinalYearProject\songs\mercy.mp3')