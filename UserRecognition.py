# -*- coding: utf-8 -*-
"""

@author: diyasinha
"""
import os 
import cv2
import numpy as np

class UserRecognition:
    def __init__(self):
        datasets = r'C:\Users\cutyp\.spyder-py3\FinalYearProject\userData'

        # Create a list of images and a list of corresponding names 
        (images, lables, self.names, id) = ([], [], {}, 0) 
        for (subdirs, dirs, files) in os.walk(datasets): 
            for subdir in dirs: 
                self.names[id] = subdir 
                subjectpath = os.path.join(datasets, subdir) 
                
                for filename in os.listdir(subjectpath): 
                    path = subjectpath + '/' + filename 
                    lable = id
                    images.append(cv2.imread(path, 0)) 
                    lables.append(int(lable)) 
                id += 1
         
  
        # Create a Numpy array from the two lists above 
        (images, lables) = [np.array(lis) for lis in [images, lables]] 
  
        # OpenCV trains a model from the images 
        self.model = cv2.face.LBPHFaceRecognizer_create() 
        self.model.train(images, lables) 
        
    def recognize_user(self, img):
        

        prediction = self.model.predict(img) 
        if prediction[1]<100: 
            return self.names[prediction[0]]

        else: 
            return 'User not recognized'