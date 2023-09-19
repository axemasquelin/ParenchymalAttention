
# Non-small Cell Lung Cancer - Parenchymal Attention
Welcome to [ParenchymalAttention](/ParenchymalAttention/). This project is designed to evaluate the level of attention a neural network places on features present in a region of interest for LDCT images. For any inquires regarding data or issues regarding running this code contact the corresponding author [Axel Masquelin](amasquelin@bwh.harvard.edu)  

## Table of Contents
- [Abstract:](#abstract)
- [Requirements:](#requirements)
- [Getting Started:](#getting-started)
- [Components:](#components)
- [Citation:](#citation)


## Abstract: 
#### BACKGROUND:
Continued improvement in deep learning methodologies has increased the rate at which deep neural networks are being evaluated for medical applications, including diagnosis of lung cancer. However, there has been limited exploration of the underlying features networks use to identify lung cancer from computed tomography (CT) images. 
#### OBJECTIVE:
In this study, we used combination of perturbation methodologies and saliency activation maps to systematically explore the contributions of both parenchymal and tumor regions in a CT image to the classification of indeterminate lung nodules.
METHODS:
We selected individuals from the National Lung Screening Trial (NLST) with solid pulmonary nodules 4 – 20 mm in diameter. Segmentation masks were used to generate three distinct datasets; 1) an Original Dataset containing the complete low-dose CT scans from the NLST, 2) a Parenchyma-Only Dataset in which the tumor regions were covered by a mask, and 3) a Tumor-Only Dataset in which only the tumor regions were included.
#### RESULTS:
The Original Dataset significantly outperformed the Parenchyma-Only Dataset and the Tumor-Only Dataset with an AUC of 81.38 ± 3.68% compared to 77.56 ± 4.42% and 77.56 ± 3.62%, respectively. Gradient-weighted class activation mapping (Grad-CAM) of the Original Dataset showed increased attention was being given to the nodule and the tumor-parenchyma boundary when nodules were classified as malignant. In the case of benign nodules, increased network attention to distant parenchymal structures, such as vasculature, emphysema, or fibrotic tissues was observed. This pattern of attention remained unchanged in the case of the Parenchyma-Only Dataset. Nodule size and first-order statistical features of the nodules were significantly different with the average malignant and benign nodule maximum 3d diameter being 23mm and 12mm, respectively. In the case of the Parenchyma-Only dataset, benign nodules were shown to be more spherical, 0.53, when compared to malignant, 0.44.
##### CONCLUSION:
We conclude that network performance is linked to textural features of nodules such as kurtosis, entropy and intensity, as well as morphological features such as sphericity and diameter. Furthermore, textural features are more positively associated with malignancy than morphologies features. 


## Requirements: 
Review [requirements.txt](/ParenchymalAttention/requirements.txt) for necessary libraries

## Getting Started:


## Components
### [bin:](/ParenchymalAttention/ParenchymalAttention/bin/)
Folder designed to contain **.sh** filetypes in order to allow rapid deployment of experiments to benchmark networks. Additionally, ```bin``` contains the following preprocessing code to generate necessary data from the NLST:  
&emsp; (1) ```cleandata.py``` matches original NLST LDCTs files with    segmentation maps. Will search for all matching PIDs and generate a csv file of all matching segmentation files and original LDCT for torch dataloader.  
&emsp; (2) ```pid_analys.py``` Provides Demographic data for PIDs present in true-positives/negatives and false-positives/negatives.   
&emsp; (3) ```statistics.py``` Conducts a Bonferroni correction on statistical analysis alongside a Welsh and Levene test.  
&emsp; (4) ```nrrdtopng.py``` Converts Nrrd files to pngs, this file is obsolet as ```/data/dataloader.py``` prioritizes csv files and tiff   files.

### [Networks:](/ParenchymalAttention/ParenchymalAttention/networks/)
Folder contains designated network architectures to evaluate model. Exisiting architecture include a ```MobileNetV1```, and a custom ```Miniception``` module. The network reported in literature is the ```Miniception``` module and is designed as a custom medical neural network. Large off the shelf DNNs do not work well for specific medical tasks and therefore custom built networks that bypass the need for transfer learning show equal or better performance (See [Transfusion: Understanding Transfer Learning for Medical Imaging paper](https://proceedings.neurips.cc/paper_files/paper/2019/file/eb1e78328c46506b46a4ac4a1e378b91-Paper.pdf) for a more comprehensive analysis of the current issues in medical datasets)

### [data:](/ParenchymalAttention/ParenchymalAttention/data/)
Contains preprocessing and dataloader functions;  
&emsp;(1) ```dataloader.py``` dataloader for medical image classification, specifically dealing with Nrrd files and image augmentation techniques.  
&emsp;(2) ```preprocess.py``` applies a requested Standardization/Normalization protocol to the images. Normalization was utilized for the reported results in manuscript. 
### [Utils:](/ParenchymalAttention/ParenchymalAttention/utils/)
Folder containing all utility function for visualization of results, saving networks, loading networks
### [results:](/ParenchymalAttention/results)
Results from experiments. 
----
## Citation:
```

```