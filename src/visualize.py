# coding: utf-8
""" MIT License """
'''
    Project: Parenchymal Attention Network
    Authors: Axel Masquelin
    Description: Separate Image Visualization p
'''
# Libraries
# --------------------------------------------
import matplotlib.pyplot as plt
from PIL import Image

import numpy as np
import cv2, os, glob, pywt
# --------------------------------------------

def otsu(img):
    """
    Definition:
    Input:
    Output:
    """
    _,thres = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)

    return thres

def nodule_remove(numparr, maskarr, mask_size):
    """
    Definition:
    Input:
    Output: 
    """
    print(numparr.shape)
    print(maskarr.shape)
    SP_x = int((im.shape[0] - mask_size)/2)                 # Defining Start X pixel, this is where the mask will start
    SP_y = int((im.shape[1] - mask_size)/2)                 # Defining Start Y pixel, this is wehere the mask will start

    for pixel_x in range(mask_size):                        # Defining x pixel location that will run through mask size
        for pixel_y in range(mask_size):                    # Defining y pixel location that will run through mask size

            if maskarr[0][pixel_x + SP_x][pixel_y + SP_y] != 0:        # If Pixel Value at start of mask location is not 0, 
                numparr[0][pixel_x + SP_x][pixel_y + SP_y] = 0    # Set the value of the new threshold image to 0

            pixel_y += 1
        pixel_x += 1
    
    return numparr[0]


def block_algo(im, mask_size):
    
    thres_img = np.zeros((1,im.shape[0],im.shape[1]))
    thres_img[0][:][:] = im                                 # Defining image that we will modify
    
    SP_x = int((im.shape[0] - mask_size)/ 2)                # Defining Start X pixel, this is where the mask will start
    SP_y = int((im.shape[1] - mask_size)/2)                 # Defining Start Y pixel, this is wehere the mask will start

    for pixel_x in range(mask_size):                        # Defining x pixel location that will run through mask size
        for pixel_y in range(mask_size):                    # Defining y pixel location that will run through mask size

            thres_img[0][pixel_x + SP_x][pixel_y + SP_y] = 0    # Set the value of the new threshold image to 0

            pixel_y += 1
        pixel_x += 1
    
    return thres_img


def normalizePlanes(nparray):
    
    data = np.zeros((1, 64, 64))

    maxHU = np.max(nparray) 
    minHU = np.min(nparray)
    
    norm = (nparray - minHU) / (maxHU - minHU)
    norm[norm>1] = 1
    norm[norm<0] = 0
    
    data[0][:][:] = norm
    
    return data

def saveImg(img, savepath, mask_size = None, name = None, ext = '.png'):
    PILimg = Image.fromarray(img)
    PILimg = PILimg.convert('RGB')
    size = (256,256)
    PILimg = PILimg.resize(size)
    if mask_size != None:
        PILimg.save(savepath + name + mask_size + ext, dpi = (600,600))
    else:
        PILimg.save(savepath + name + ext, dpi = (600,600))
    PILimg.close()

if __name__ == '__main__':  
    '''Init Variables'''
    mask_size = 16
    
    '''Paths'''
    imagespath = os.path.split(os.getcwd())[0] + '/dataset/Group2/*.jpg'
    resultpath = os.path.split(os.getcwd())[0] + r'/results/NoduleImg/'
    img_list = glob.glob(imagespath)
    print(img_list)
    numparr = np.zeros((1,64,64))
    maskarr = np.zeros((1,64,64))

    for n in range(len(img_list)):
        print(img_list[n])
        im = cv2.imread(img_list[n], 0)
        
        numparr[0][:][:] = im
        otsu_im = otsu(im)    
    
        maskarr[0][:][:] = otsu_im  
        mask_im = nodule_remove(numparr, maskarr, mask_size)

        # saveImg(im, resultpath, name = 'Original')
        # saveImg(otsu_im, resultpath, name = 'Otsu')
        # saveImg(mask_im, resultpath, str(mask_size), name = "OtsuMask")

        cv2.imshow('images', im)
        # cv2.imshow('Otsu', otsu_im)        

        plt.show()
        cv2.waitKey(0)
