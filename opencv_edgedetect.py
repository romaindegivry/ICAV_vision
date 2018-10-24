# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 02:00:16 2018

@author: user
"""

import cv2
import numpy as np

cap = cv2.VideoCapture(0)

#cap = cv2.imread('C:\\Users\\user\\Desktop\\cat.jpg')

while True:

    __, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #defining colors using hsv


    laplacian = cv2.Laplacian(frame, cv2.CV_64F)
    sobelx = cv2.Sobel(frame, cv2.CV_64F,1,0, ksize=5)
    #                                   x  y
    sobely = cv2.Sobel(frame, cv2.CV_64F,0,1, ksize=5)
    edges = cv2.Canny(frame, 50,50) #changing size changes amount of noise

    cv2.imshow('original',frame)
    cv2.imshow('laplacian', laplacian)
    cv2.imshow('sobelx', sobelx)
    cv2.imshow('sobely', sobely)
    cv2.imshow('edges', edges)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()