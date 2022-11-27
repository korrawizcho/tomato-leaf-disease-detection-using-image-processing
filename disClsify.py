import cv2
import os
import math
import numpy as np
import random
import skimage.feature as feature
from scipy.stats import skew



def _Equalization(img):
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)    
    return img_output

def _RemoveBackground(img):
    ## convert to hsv
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # mask image within range
    mask = cv2.inRange(hsv, (10, 20, 20), (110, 255,220))
    
    # reduce noise using morphological transform
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20,20))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # convering mask to bool mask
    imask = mask>0
    # masking image
    no_bg_image = np.zeros_like(img, np.uint8)
    no_bg_image[imask] = img[imask]

    return no_bg_image, mask 

def _GradientMagnitude(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # finding gradient magnitude using sobel x and y
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_x = cv2.convertScaleAbs(grad_x)
    grad_y = cv2.convertScaleAbs(grad_y)
    
    return [np.mean(grad_x),
            np.std(grad_x),
            np.mean(grad_y), 
            np.std(grad_y)]

def _ContourFeatures(mask):
    # apply morphology for removing some noise 
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25,25))
    morph = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    edged = cv2.Canny(morph, 30, 200)
    
    contours,_ = cv2.findContours(edged, 
    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    # get largest contour
    area_list = np.array([cv2.contourArea(cnt) for cnt in contours])
    
    # avoid error when there is no contour
    try:
        largest_cnt = contours[np.argmax(area_list)]
        area = np.amax(area_list)
        x,y,w,h = cv2.boundingRect(largest_cnt)
        (cenx,ceny),radius = cv2.minEnclosingCircle(largest_cnt)
        M = cv2.moments(largest_cnt)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00']) 
        
    except:
        w,h,cx,cy,radius = [0,0,0,0,0]
        
    return [w,h,cx,cy,radius]

def _GLCM(gray):

    graycom = feature.graycomatrix(gray, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256)
    # Get the GLCM properties
    contrast = feature.graycoprops(graycom, 'contrast')
    dissimilarity = feature.graycoprops(graycom, 'dissimilarity')
    homogeneity = feature.graycoprops(graycom, 'homogeneity')
    energy = feature.graycoprops(graycom, 'energy')
    correlation = feature.graycoprops(graycom, 'correlation')
    ASM = feature.graycoprops(graycom, 'ASM')
    
    return [dissimilarity.tolist()[0], homogeneity.tolist()[0], 
            energy.tolist()[0], correlation.tolist()[0],
            contrast.tolist()[0], ASM.tolist()[0]]
    
def _TextureFeatures(img):
    # finding texture features by GLCM method
    gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
    G = _GLCM(gray)
    G_list = [round(np.mean(featrs),4) for featrs in G ]
    
    return [val for val in G_list]

def _ColorFeatures(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # choosing mean , std, correlation, skewness from 3 channel
    mean = [ np.mean(hsv[...,ch]) for ch in range(3)]
    std = [ np.std(img[...,ch]) for ch in range(3)]
    corr = [np.correlate(hsv[...,0].flatten(), hsv[...,1].flatten()),
            np.correlate(hsv[...,0].flatten(), hsv[...,2].flatten()),]
    kswness = [skew(img[...,ch].flatten()) for ch in range(3)]
    
    return mean + std + corr + kswness

def _KeypointFeatures(img):
    img = img.copy()
    
    # find keypoints
    orb = cv2.ORB_create(200)
    kps = orb.detect(img,None)
    kps, des = orb.compute(img, kps)
    
    # calculate mean and std of keypoints
    stat_des = [np.mean(des) , np.std(des)]    

    return stat_des

def _FeaturesExtraction(image_path):
    
            image = cv2.imread(image_path)
            # reduceing noise using gaussian blur
            image = cv2.GaussianBlur(image,(5,5),0)
            
            # using histogram as a feature
            hist = cv2.calcHist([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY 
                                              )],[0],None,[100],[0,256])
            hist = list(hist.flatten())
            
            #remove background
            no_bg, mask =  _RemoveBackground(image)
            # improve image quality by equlize the histogram
            no_bg = _Equalization(no_bg)
            
            grad_features = _GradientMagnitude(no_bg)
            cnt_features= _ContourFeatures(mask)
            text_features = _TextureFeatures(no_bg)
            color_features = _ColorFeatures(no_bg)
            keypoints_features = _KeypointFeatures(no_bg)
            
            features_list = text_features + hist + color_features\
                + cnt_features+ grad_features + keypoints_features

            return features_list
    
