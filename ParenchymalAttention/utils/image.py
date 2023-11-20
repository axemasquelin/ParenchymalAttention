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
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import torch
import os, cv2
import ParenchymalAttention.data.dataloader as dataloader
import ParenchymalAttention.utils.utils as utils
# ---------------------------------------------------------------------------- #

class CAM:
    """
    Implementation of Class Activation Map
    """
    def __init__(self, model, target_layer):
        self.model = model.eval()
        self.target_layer = target_layer

    def forward(self, x):
        return self.model(x)

    def __call__(self, x, index=None):
        output = self.forward(x)
        if index is None:
            index = np.argmax(output.cpu().data.numpy())
        target = output[0][index]

        self.model.zero_grad()
        conv_output, target_grad = self.get_conv_output_grad(x, target)
        activations = self.get_activations(conv_output)
        weights = self.get_weights(target_grad, activations)
        cam = self.get_cam(activations, weights)
        cam = cv2.resize(cam, x.shape[2:])
        cam = cam - np.min(cam)
        cam = cam

class GradCAM:
    """
    Implementation of GradCAM methodology to visualize features of importance in a given class
        Uses: Gradient + Feature Map
    """
    def __init__(self, model):
        self.model = model
        self.feature_maps = None
        self.gradient = None
        self.forward_hook = None
        self.backward_hook = None

    def forward_hook_fn(self, module, input, output):
        """
        Function to gather feature map from select convolution layer
        -----------
        Parameters:
        module - 
        input - 
        output - 
        """
        self.feature_maps = output.detach()

    def backward_hook_fn(self, module, grad_in, grad_out):
        """
        Backward Hook for gradients and to detatch them from tensor
        -----------
        Parameters:
        """
        self.gradient = grad_out[0].detach()

    def get_gradients(self, input_tensor, target_class):
        """
        Function will get the backward gradient of the network by passing an input image for a select target class
        -----------
        Parameters:
        input_tensor - Tensor
            Image from samples
        target_class - Tensor
            Class label of interest [0 - Benign], [1 - Malignant]
        --------
        Returns:
        gradient -
        """
        self.model.zero_grad()
        output = self.model(input_tensor)
        output[0, target_class].backward()
        return self.gradient

    def generate_cam(self, input_tensor, target_class):
        """
        Generat Grad camera of specific layer and needs to be converted to heat map
        -----------
        Parameters:
        Returns:
        """
        # self.forward_hook = self.model.lastconv.register_forward_hook(self.forward_hook_fn)
        # self.backward_hook = self.model.lastconv.register_backward_hook(self.backward_hook_fn)
        
        self.forward_hook = self.model.layers.register_forward_hook(self.forward_hook_fn)
        self.backward_hook = self.model.layers.register_backward_hook(self.backward_hook_fn)

        gradients = self.get_gradients(input_tensor.unsqueeze(0), target_class)
        alpha = gradients.mean(dim=(2, 3), keepdim=True)
        
        weighted_feature_maps = (alpha * self.feature_maps).sum(dim=1, keepdim=True)
        
        cam = F.relu(weighted_feature_maps)
        cam = F.interpolate(cam, size=input_tensor.shape[-2:], mode='bilinear', align_corners=False)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-7)
        
        self.forward_hook.remove()
        self.backward_hook.remove()

        return cam.squeeze().cpu().numpy()


def getcam(testloader, net, method, fold:int, rep:int, masksize=None, selectcam:str='GradCAM', device=None, folder='/StandardImages/', normalize=True):
    """"
    Description:
    -----------
    Parameters:
    --------
    Returns:
    """
    # net = torch.load(os.getcwd() + '/results/' + method + '/' + method +'_bestnetwork.pt')
    utils.create_directories(folder= f'/results/{method}/GradCAM/{str(fold)}_{str(rep)}/FN')
    utils.create_directories(folder= f'/results/{method}/GradCAM/{str(fold)}_{str(rep)}/TN')
    utils.create_directories(folder= f'/results/{method}/GradCAM/{str(fold)}_{str(rep)}/FP')
    utils.create_directories(folder= f'/results/{method}/GradCAM/{str(fold)}_{str(rep)}/TP')

    if selectcam=='GradCAM':
        cam_method = GradCAM(model=net)

    for i, data in enumerate(testloader):        
        images = data['image'].to(device=device, dtype=torch.float)
        labels = data['label'].to(device=device)
        pids = data['id']
        times = data['time']
    
        output = net(images)
        con, pred = torch.max(output, 1)
        for i, image in enumerate(images):
            ca = labels[i].detach().cpu().squeeze()
            pid = pids[i]       
            time = times[i]
            cam = cam_method.generate_cam(image, target_class=ca)
            confidence = con[i].detach().cpu().numpy()
            prediction = pred[i].detach().cpu().squeeze()

            if ((prediction == 1) and (prediction == ca)):
                fileid = str(pid) + '_' + str(ca.numpy()) + '_' + str(time) + '_TP'
                if folder != '/StandardImages/':
                    folderpath = folder + f'{str(fold)}_{str(rep)}/TP/'
            elif ((prediction == 1) and (prediction != ca)):
                fileid = str(pid) + '_' + str(ca.numpy()) + '_' + str(time) + '_FP'
                if folder != '/StandardImages/':
                    folderpath = folder + f'{str(fold)}_{str(rep)}/FP/'
            elif ((prediction == 0) and (prediction == ca)):
                fileid = str(pid) + '_' + str(ca.numpy()) + '_' + str(time) + '_TN'
                if folder != '/StandardImages/':
                    folderpath = folder + f'{str(fold)}_{str(rep)}/TN/'
            elif ((prediction == 0) and (prediction != ca)):
                fileid = str(pid) + '_' + str(ca.numpy()) + '_' + str(time) + '_FN'
                if folder != '/StandardImages/':
                    folderpath = folder + f'{str(fold)}_{str(rep)}/FN/'
            
            saveImageCam(image, cam, confidence, ca, prediction, method, ms=masksize,title= fileid,folder=folderpath)

def saveImageCam(image, cam, confidence, label, pred, method, ms = None, title=None, folder = '/StandardImages/'):
    """"
    Description:
    -----------
    Parameters:
    --------
    Returns:
    """
    image = image.cpu().squeeze()
    im = np.zeros((image.shape[0],image.shape[1],3))
    im[:,:,0] = image
    im[:,:,1] = image
    im[:,:,2] = image

    # Convert the image to a heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_VIRIDIS)
    heatmap = heatmap/np.max(heatmap)
    
    # Overlay the heatmap onto the original image
    overlay = heatmap * 0.5 + np.float32(im)
    overlay *= 255/np.amax(overlay)
    overlay = np.uint8(overlay)
    overlay_plt = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    resultpath = os.getcwd() + "/results/"
    text_conf = 'Confidence: ' + str(np.round(confidence,2))
    text_class = 'Prediction: ' + str(pred.numpy()) + ' | Actual: ' + str(label.numpy())

    plt.figure()
    ax = plt.subplot(111)
    ax.set_clip_on(False)
    plt.plot([0,1],[0,1])
    plt.imshow(overlay_plt, vmin = np.min(overlay_plt) , vmax = np.max(overlay_plt))
    plt.colorbar()
    plt.axis('off')
    ax.annotate(text_conf, xy=(0,0),xytext=(0,-2))
    ax.annotate(text_class, xy=(0,0),xytext=(0,-6))

    if ms != None:
        plt.savefig(resultpath + method  + str(ms) + 'x' + folder + title + '_PLTattention.png')
        cv2.imwrite(resultpath + method + str(ms) + 'x' + folder + title + '_attention.png', overlay)

    else:
        plt.savefig(resultpath + method + folder + title + '_PLTattention.png')
        cv2.imwrite(resultpath + method + folder + title + '_attention.png', overlay)
        
    plt.close()