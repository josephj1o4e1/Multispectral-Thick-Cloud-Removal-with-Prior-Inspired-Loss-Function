# Multispectral Thick Cloud Removal with Prior-Inspired Loss Function
This project is about removing cloud and haze obstacles from satellite images using deep learning frameworks.  
These satellite images all are 16-bit depth, 4 channel RGB-NIR images, where the pixel values represent raw values of Level 4 Top of atmosphere
reflectance (TOAR).  
We tested out a new idea of a NDVI/NDWI loss function, which uses geoscience metrics as a loss function that gives it a cross channel color stabilizaiton. This remains to be tested thoroughly. Here is an example of our result: 

Before / After (Satellite Image 4 channel TOAR):  
![image](https://github.com/josephj1o4e1/Multispectral-Thick-Cloud-Removal-with-Prior-Inspired-Loss-Function/assets/13396370/aa969532-2a0d-43bb-9a45-6f1bc69b1bc1)
  
Before / After (NDVI Heat Map):  
![image](https://github.com/josephj1o4e1/Multispectral-Thick-Cloud-Removal-with-Prior-Inspired-Loss-Function/assets/13396370/bade9d02-904c-4d11-8724-cba05f6ccd9d)


Before / After (Satellite Image 4 channel TOAR):  
![image](https://github.com/josephj1o4e1/Multispectral-Thick-Cloud-Removal-with-Prior-Inspired-Loss-Function/assets/13396370/b9805187-c42e-4527-84fe-980688231fe3)
  
Before / After (NDVI Heat Map):  
![image](https://github.com/josephj1o4e1/Multispectral-Thick-Cloud-Removal-with-Prior-Inspired-Loss-Function/assets/13396370/6cbd7c9d-4d88-491e-9c83-383452be0ec7)
  

Here is our overall architecture:  
  
![architecture](/images/System_Cloud_Removal_with_NDX_Loss.png)

## How to use this code?
Please use the shell script main provided "main_trainNDVI_NDWI_s2.sh".  
Command could be like:  
sh main_trainNDVI_NDWI_s2.sh 0 L1C_England_augmented 0 80 0.0001 8 NTIRE2018_Multiscale_NDVI_NDWI 512
The argument L1C_England_augmented is the dataset pathname of your own dataset. 
