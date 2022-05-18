# Multispectral Thick Cloud Removal with Prior-Inspired Loss Function
This project is about removing cloud and haze obstacles from satellite images using deep learning frameworks.  
These satellite images all are 4 channel RGB-NIR images with 
We tested out a new idea of a NDVI/NDWI loss function, which uses geoscience metrics as a loss function that gives it a cross channel color stabilizaiton. This remains to be tested thoroughly. Here is an example of our result:  
  
![result](/images/cloud_decloud.png)

Here is our overall architecture:  
  
![architecture](/images/System_Cloud_Removal_with_NDX_Loss.png)

## How to use this code?
