# LHC-Net
## Local Multi-Head Channel Self-Attention

This repository is intended to provide a quick implementation of the LHC-Net and to replicate the results in this [paper](https://www.dropbox.com/s/ltqykplbjk6ks3g/Rev4.pdf?dl=1) on FER2013 by downloading our trained models or, when hardware compatibility is available, by training the model from scratch. A fully custom training routine is also available.

![Image of LHC_Net](https://github.com/Bodhis4ttva/LHC_Net/blob/main/Images/LHC_Net.jpg)
![Image of LHC_Module2](https://github.com/Bodhis4ttva/LHC_Net/blob/main/Images/LHC_Module2.jpg)

## How to:<br />
Clone the repository and execute main.py

**Requirements for full replicability: <br />**
Nvidia Geforce GTX-1080ti (any Pascal GPU should work)<br />
Driver 457.51 <br />
Cuda 11.1.1* <br />
CuDNN v8.0.5 - 11.1 <br />
Python 3.8.5 <br />
requirements.txt
<br />
<br />
*After Cuda installation rename C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.1\bin\cusolver64_11.dll in ...\cusolver64_10.dll

**Note that:<br />**
Executing main.py will automatically download the data from dropbox and create a folder with the entire FER2013 training set saved as jpeg
