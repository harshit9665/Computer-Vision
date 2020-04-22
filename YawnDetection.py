# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 13:45:22 2020

@author: diyasinha
"""
import cv2
import dlib
import numpy as np

class YawnDetection:
    def __init__(self):
        PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
        self.predictor = dlib.shape_predictor(PREDICTOR_PATH)

        self.detector = dlib.get_frontal_face_detector()
        
    def get_landmarks(self,im):
        rects = self.detector(im, 1)
        s = "err"
        if len(rects) > 1:
            return s
        if len(rects) == 0:
            return s
        return np.matrix([[p.x, p.y] for p in self.predictor(im, rects[0]).parts()])
    
    def annotate_landmarks(self,im, landmarks):
        im = im.copy()
        for idx, point in enumerate(landmarks):
            pos = (point[0, 0], point[0, 1])
            cv2.putText(im, str(idx), pos,
                        fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                        fontScale=0.4,
                        color=(0, 0, 255))
            cv2.circle(im, pos, 3, color=(0, 255, 255))
        return im
    def top_lip(self,landmarks):
        top_lip_pts = []
        for i in range(50,53):
            top_lip_pts.append(landmarks[i])
        for i in range(61,64):
            top_lip_pts.append(landmarks[i])
        #top_lip_all_pts = np.squeeze(np.asarray(top_lip_pts))
        top_lip_mean = np.mean(top_lip_pts, axis=0)
        return int(top_lip_mean[:,1])
    
    def bottom_lip(self,landmarks):
        bottom_lip_pts = []
        for i in range(65,68):
            bottom_lip_pts.append(landmarks[i])
        for i in range(56,59):
            bottom_lip_pts.append(landmarks[i])
        #bottom_lip_all_pts = np.squeeze(np.asarray(bottom_lip_pts))
        bottom_lip_mean = np.mean(bottom_lip_pts, axis=0)
        return int(bottom_lip_mean[:,1])

    def mouth_open(self,image):
        landmarks = self.get_landmarks(image)
        
        if landmarks == "err":
            return image, 0
        
        image_with_landmarks = self.annotate_landmarks(image, landmarks)
        top_lip_center = self.top_lip(landmarks)
        bottom_lip_center = self.bottom_lip(landmarks)
        lip_distance = abs(top_lip_center - bottom_lip_center)
        return image_with_landmarks, lip_distance
    
    def predict(self,im):
        image_landmarks, lip_distance = self.mouth_open(im)
        yawn_status = 0
    
        if lip_distance > 25:
            yawn_status = 1 
        return yawn_status