# coding: utf-8
""" MIT License """
'''
    Project: Parenchymal Attention Network
    Authors: Axel Masquelin
    Description: Image manipulation and visualization function
'''
# Libraries
# ---------------------------------------------------------------------------- #
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import torch
import os, cv2
import ParenchymalAttention.data.dataloader as dataloader
# ---------------------------------------------------------------------------- #
def otsu_algo(im, mask_size):
    """"
    Description:
    -----------
    Parameters:
    --------
    Returns:
    """

    thres_img = np.zeros((1,im.shape[0],im.shape[1]))
    mask = np.zeros((1,im.shape[0],im.shape[1]))

    _,thresh = cv2.threshold(im, 127, 255, cv2.THRESH_OTSU)         # Generating Otsu Threshold Map
    thres_img[0][:][:] = im                                         # Defining image that we will modify
    mask[0][:][:] = thresh                                          # Setting mask equal to Otsu Threshold output
    
    SP_x = int((im.shape[0] - mask_size)/ 2)                        # Defining Start X pixel, this is where the mask will start
    SP_y = int((im.shape[1] - mask_size)/2)                         # Defining Start Y pixel, this is wehere the mask will start

    for pixel_x in range(mask_size):                                # Defining x pixel location that will run through mask size
        for pixel_y in range(mask_size):                            # Defining y pixel location that will run through mask size

            if mask[0][pixel_x + SP_x][pixel_y + SP_y] != 0:        # If Pixel Value at start of mask location is not 0, 
                thres_img[0][pixel_x + SP_x][pixel_y + SP_y] = 0    # Set the value of the new threshold image to 0

            pixel_y += 1
        pixel_x += 1
    
    return thres_img

def block_algo(im, mask_size):
    """"
    Description:
    -----------
    Parameters:
    --------
    Returns:
    """
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

def segmentation_map(im, maskmap):
    """"
    Description:
    -----------
    Parameters:
    --------
    Returns:
    """
    # print(im.shape)
    thres_img = np.zeros(im.shape)
    # print(thres_img.shape)
    for i in range(im.shape[0]):
        thres_img[i,:,:] = im[i,:,:] * maskmap[0]

    return thres_img
    
def getcam(testloader, masksize, net, method, device, folder = '/StandardImages/', normalize=True):
    """"
    Description:
    -----------
    Parameters:
    --------
    Returns:
    """

    for i, data in enumerate(testloader):
        images = data['image'].to(device = device, dtype = torch.float)
        label = data['label'].to(device = device)
        pids = data['id']

        
        output, x_avgs = net(images, flag=True)

        _, pred = torch.max(output, 1)

        for i in range(len(label)):
            image = images[i].cpu().numpy()
            prediction = pred[i].cpu().squeeze()
            pid = pids[i]
            ca = label[i].cpu().squeeze()

            if ((prediction == 1) and (prediction == ca)):
                fileid = str(pid) + '_TP'
                if folder != '/StandardImages/':
                    folder += 'TP/'
            elif ((prediction == 1) and (prediction != ca)):
                fileid = str(pid) + '_FP'
                if folder != '/StandardImages/':
                    folder += 'FP/'
            elif ((prediction == 0) and (prediction == ca)):
                fileid = str(pid) + '_TN'
                if folder != '/StandardImages/':
                    folder += 'TN/'
            elif ((prediction == 0) and (prediction != ca)):
                fileid = str(pid) + '_FN'
                if folder != '/StandardImages/':
                    folder += 'FN/'
            
            x_avg = x_avgs[i].cpu().detach().numpy()
            showImageCam(image, x_avg, prediction, net, method, ms = masksize, title = fileid, folder = folder)

def showImageCam(img, mask, pred, net, method, ms = None, title=None, folder = '/StandardImages/'):
    """"
    Description:
    -----------
    Parameters:
    --------
    Returns:
    """
    nc, h, w, = mask.shape
    depth, width, height = img.shape

    params = list(net.parameters())
    weight_softmax = np.squeeze(params[-2].cpu().data.numpy())

    if pred == 0:
        cam = weight_softmax[0].dot(mask.reshape((nc, h*w)))
    else:
        cam = weight_softmax[1].dot(mask.reshape((nc, h*w)))

    cam = cam.reshape(w, h)
    cam_img = (cam - np.min(cam))/(np.max(cam) - np.min(cam))
    cam_img = np.uint8(255 * cam_img)
    cam_img = cv2.resize(cam_img, (64,64))
    
    heatmap = cv2.applyColorMap(cv2.resize(cam_img, (width, height)), cv2.COLORMAP_VIRIDIS)
    
    im = np.zeros((64,64,3))
    im[:,:,0] = img[0][0]
    im[:,:,1] = img[0][0]
    im[:,:,2] = img[0][0]

    img_heat = heatmap*0.7 + im

    resultpath = os.getcwd() + "/results/"

    img_heat *= 255.0/np.amax(img_heat)
    img_heat = np.uint8(img_heat)
    img_heat_plt = cv2.cvtColor(img_heat, cv2.COLOR_BGR2RGB)
    # img_heat = np.interp(img_heat(np.amin(img_heat), np.amax(img_heat)), (0,255))
    
    plt.imshow(img_heat_plt, vmin = 150 , vmax = 255)
    plt.colorbar()
    
    if ms != None:
        plt.savefig(resultpath + method  + str(ms) + 'x' + folder + title + '_attention.png')
        # cv2.imwrite(resultpath + method + str(ms) + 'x' + folder + title + '_attention.png', img_heat)    
        cv2.imwrite(resultpath + method + str(ms) + 'x' + folder + title + '_original.png', im)

    else:
        plt.savefig(resultpath + method + folder + title + '_attention.png')
        # cv2.imwrite(resultpath + method + folder + title + '_attention.png', img_heat)
        cv2.imwrite(resultpath + method + folder + title + '_original.png', im*125)
    
    plt.close()

     