# coding: utf-8
""" MIT License """
'''
    Project: Parenchymal Attention Network
    Authors: Axel Masquelin
    Description: Image manipulation and visualization function
'''
# Libraries
# ---------------------------------------------------------------------------- #
from torchvision.transforms import Compose, Normalize, ToTensor
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import dataload
import torch
import os, cv2
# ---------------------------------------------------------------------------- #
def otsu_algo(im, mask_size):
    
    thres_img = np.zeros((1,im.shape[0],im.shape[1]))
    mask = np.zeros((1,im.shape[0],im.shape[1]))

    _,thresh = cv2.threshold(im, 127, 255, cv2.THRESH_OTSU) # Generating Otsu Threshold Map
    thres_img[0][:][:] = im                                 # Defining image that we will modify
    mask[0][:][:] = thresh                                  # Setting mask equal to Otsu Threshold output
    
    SP_x = int((im.shape[0] - mask_size)/ 2)                # Defining Start X pixel, this is where the mask will start
    SP_y = int((im.shape[1] - mask_size)/2)                 # Defining Start Y pixel, this is wehere the mask will start

    for pixel_x in range(mask_size):                                # Defining x pixel location that will run through mask size
        for pixel_y in range(mask_size):                            # Defining y pixel location that will run through mask size

            if mask[0][pixel_x + SP_x][pixel_y + SP_y] != 0:        # If Pixel Value at start of mask location is not 0, 
                thres_img[0][pixel_x + SP_x][pixel_y + SP_y] = 0    # Set the value of the new threshold image to 0

            pixel_y += 1
        pixel_x += 1
    
    return thres_img

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

def getcam(filepath, masksize, net, dataset, device, fileid = None, folder = '/StandardImages/', normalize=True):
    """
    Definition:
    Input:
        -net: Network variable
    """

    load_im = np.zeros((1, 64, 64))
    ca = os.path.splitext(filepath.split('_')[-1])[0]
    ca = ca.split('.')[0]
    label = []

    if ((ca == '1') or (ca == '0')):        
        if ca == '1':
            label.append(1)
        else: 
            label.append(0)

        img = cv2.imread(filepath, 0)

        load_im[0, :, :] = np.array(img)

        if dataset != 'Original':
            load_im = dataload.nodule_remove(img, masksize, normalize = True, masktype = dataset)

        im = np.zeros((1,1,64,64))
        im[0,:,:,:] = load_im

        torch_im = torch.from_numpy(im)
        torch_label = torch.from_numpy(np.asarray(label))

        image = torch_im.to(device=device, dtype=torch.float)
        label = torch_label.to(device=device)
        
        output, x_avg = net(image, flag=True)

        _, pred = torch.max(output, 1)

        if ((pred == 1) and (pred == label)):
            fileid = fileid.split('.')[0] + '_TP'
            if folder != '/StandardImages/':
                folder += 'TP/'
        elif ((pred == 1) and (pred != label)):
            fileid = fileid.split('.')[0] + '_FP'
            if folder != '/StandardImages/':
                folder += 'FP/'
        elif ((pred == 0) and (pred == label)):
            fileid = fileid.split('.')[0] + '_TN'
            if folder != '/StandardImages/':
                folder += 'TN/'
        elif ((pred == 0) and (pred != label)):
            fileid = fileid.split('.')[0] + '_FN'
            if folder != '/StandardImages/':
                folder += 'FN/'
        
        x_avg = x_avg.cpu().detach().numpy()

        showImageCam(im, x_avg, label, net, dataset, ms = masksize, title = fileid, folder = folder)

def showImageCam(img, mask, label, net, dataset, ms = None, title=None, folder = '/StandardImages/'):
    """
    Definition:
    Input:
        -img : Image 
        -mask: Mask of Network Attention
    Output: 
        -img_heat: Image with Heatmap to show network attention. 
    """
    batch, nc, h, w, = mask.shape
    width, height = img[0][0].shape

    params = list(net.parameters())
    weight_softmax = np.squeeze(params[-2].cpu().data.numpy())

    if label == 0:
        cam = weight_softmax[0].dot(mask.reshape((nc, h*w)))
    else:
        cam = weight_softmax[1].dot(mask.reshape((nc, h*w)))

    cam = cam.reshape(w, h)
    cam_img = (cam - np.min(cam))/(np.max(cam) - np.min(cam))
    cam_img = np.uint8(255 * cam_img)
    cam_img = cv2.resize(cam_img, (64,64))
    
    heatmap = cv2.applyColorMap(cv2.resize(cam_img, (width, height)), cv2.COLORMAP_JET)
    
    im = np.zeros((64,64,3))
    im[:,:,0] = img[0][0]
    im[:,:,1] = img[0][0]
    im[:,:,2] = img[0][0]

    img_heat = heatmap*0.7 + im

    resultpath = os.path.split(os.getcwd())[0] + "/results/"

    img_heat *= 255.0/np.amax(img_heat)
    img_heat = np.uint8(img_heat)
    img_heat_plt = cv2.cvtColor(img_heat, cv2.COLOR_BGR2RGB)
    # img_heat = np.interp(img_heat(np.amin(img_heat), np.amax(img_heat)), (0,255))
    
    # plt.imshow(img_heat, vmin = 150 , vmax = 255)
    # plt.colorbar()
    
    if dataset != "Original":
        cv2.imwrite(resultpath + dataset + '/' + str(ms) + 'x' + folder + title + '_attention.png', img_heat)    
        cv2.imwrite(resultpath + dataset + '/' + str(ms) + 'x' + folder + title + '_original.png', im)

    else:

        cv2.imwrite(resultpath + dataset + folder + title + '_attention.png', img_heat)
        cv2.imwrite(resultpath + dataset + folder + title + '_original.png', im)

     