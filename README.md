
# Non-small Cell Lung Cancer - Parenchymal Attention
Welcome to [ParenchymalAttention](/ParenchymalAttention/). This project is designed to evaluate the level of attention a neural network places on features present in a region of interest for LDCT images. For any inquires regarding data or issues regarding running this code contact the corresponding author [Axel Masquelin](amasquelin@bwh.harvard.edu)  

## Scratch Branch:
This branch is designed as a playground for further improving this project post publication. 

## Table of Contents
- [Requirements:](#requirements)
- [Getting Started:](#getting-started)
- [Components:](#components)
- [Citation:](#citation)


## Requirements: 
Review [requirements.txt](/ParenchymalAttention/requirements.txt) for necessary libraries

## Getting Started:


## Components
### [bin:](/ParenchymalAttention/ParenchymalAttention/bin/)
Folder designed to contain **.sh** filetypes in order to allow rapid deployment of experiments to benchmark networks. Additionally, ```bin``` contains the following preprocessing code to generate necessary data from the NLST:  
&emsp; (1) ```cleandata.py``` matches original NLST LDCTs files with    segmentation maps. Will search for all matching PIDs and generate a csv file of all matching segmentation files and original LDCT for torch dataloader.  
&emsp; (2) ```pid_analysis.py``` Provides Demographic data for PIDs present in true-positives/negatives and false-positives/negatives.   
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
----
## Citation:
```

```
