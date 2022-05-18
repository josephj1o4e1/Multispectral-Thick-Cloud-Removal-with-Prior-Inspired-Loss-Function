import os
import sys
import argparse
import time
import re
import numpy as np
from PIL import Image
from natsort import natsorted, ns

import rasterio

# python3 noresize_crop_shift_s2.py --datasetname 20210303_crop30/val/Hazy --L 512
# if it is read from not-combined, BGR-NIR->RGB-NIR
# if it is read from combined, RGB-NIR->BGR-NIR->RGB-NIR

CURR_DIR = os.path.abspath(os.path.dirname(__file__)) #C:/.../Indoor
parser = argparse.ArgumentParser()
parser.add_argument('--datasetname', required=False,
  default='20210303_crop30/val/Hazy',  help='')
parser.add_argument('--format', required=False,
  default='tiff',  help='it is png, tiff, jpg')
parser.add_argument('--L', required=False, type=int,
  default=512,  help='the length of the square crop patch')
parser.add_argument('--combined', required=False, type=int,
  default=1,  help='usually deal with combined data, sometimes deal with split band raw data, then switch assign 0. ')
opt = parser.parse_args()

def crop(rast, box):
  x0,x1,y0,y1 = map(int, map(round, box)) # THE MAIN REASON WHY IT IS NO-RESIZE!
  return rast[y0:y1,x0:x1,:] # H,W,C ---> y,x,...

datasetname=''
for i in range(len(re.split('/',opt.datasetname))):
  if i<len(re.split('/',opt.datasetname))-1:
    datasetname = datasetname + re.split('/',opt.datasetname)[i] + '_'
  else:
    datasetname = datasetname + re.split('/',opt.datasetname)[i]

unprocessed_dir = os.path.join(CURR_DIR, 'dataset', opt.datasetname) 
if not os.path.exists(unprocessed_dir):
    err='unprocessed_dir path:' + unprocessed_dir + 'doesn\'t exist!'
    sys.exit(err)

processed_dir = os.path.join(CURR_DIR, 'image_results', 'noresize_crop_shift', datasetname, 'patch'+str(opt.L)) # .../image_results/noresize_crop_shift/20210303_val/patch512
# processed_dir = os.path.join(CURR_DIR, 'image_results', 'noresize_crop_shift', datasetname+'gt', 'patch'+str(opt.L)) # .../image_results/noresize_crop_shift/20210303_val/patch512
if not os.path.exists(processed_dir):
  os.makedirs(processed_dir)
if os.listdir(processed_dir):
  print("ALREADY CROPPED BEFORE!!")
  sys.exit()

# READ DATA
roots=[]
jp2_fnames=[]
rasters_hazy=[]
if opt.combined==0: # combine raw split band s2 data.....IF WE WANT TO CROP GT, this part needs to add: "if (rt.endswith("/GT")):"...blahblahblah
    for root, _, fnames in (os.walk(unprocessed_dir)):
        for fname in fnames:
            if (fname.endswith(".jp2")):
                roots.append(root) 
                jp2_fnames.append(fname)
    for i, (rt, f) in enumerate(zip(roots, jp2_fnames)):
        if (rt.endswith("/hazy")):
        # if (rt.endswith("/GT")):
            print(os.path.join(rt, f))
            if(f.endswith("B02_10m.jp2") or f.endswith("B02.jp2")):
                raster1 = rasterio.open(os.path.join(rt, f))
                filename=f.replace("_B02_10m.jp2", "")
            elif(f.endswith("B03_10m.jp2") or f.endswith("B03.jp2")):
                raster2 = rasterio.open(os.path.join(rt, f))
            elif(f.endswith("B04_10m.jp2") or f.endswith("B04.jp2")):
                raster3 = rasterio.open(os.path.join(rt, f))
            elif(f.endswith("B08_10m.jp2") or f.endswith("B08.jp2")):
                raster4 = rasterio.open(os.path.join(rt, f))
            else: break
            
            if i%4==3: # should use a different variable to count instead of i
                band1 = raster1.read(1) # B (sentinel-2 is read as BGR-NIR from raw data rasterio)
                band2 = raster2.read(1) # G
                band3 = raster3.read(1) # R
                band4 = raster4.read(1) # NIR
                raster_combined = np.dstack((band1,band2,band3,band4))            
                CRS=raster4.crs
                Transform=raster4.transform
                dType=raster4.dtypes[0]
                print("combinedhazy...")
                rasters_hazy.append((raster_combined,filename))

else: # already combined beforehand
    for root, _, fnames in (os.walk(unprocessed_dir)):
        for i, fname in enumerate( natsorted(fnames, key=lambda y: y.lower()) ):
            raster = rasterio.open(os.path.join(unprocessed_dir, fname))
            CRS=raster.crs
            Transform=raster.transform
            dType=raster.dtypes[0]
            band3 = raster.read(1) # R (sentinel-2 is read as RGB-NIR from after crop30, but for convenience, we still flip it back for the final combine to be written simple.)
            band2 = raster.read(2) # G
            band1 = raster.read(3) # B
            band4 = raster.read(4) # NIR
            raster_combined = np.dstack((band1,band2,band3,band4))
            filename = fname.split('.')[0]
            rasters_hazy.append((raster_combined, filename))


# CROP combined band s2 data 
L = opt.L
output_w = L
output_h = L
for i, (img, imgname) in enumerate(rasters_hazy): 
    t0 = time.time()
    H, W = img.shape[0], img.shape[1]
    if(H <= L):
      crop_countH = 1
    else:
      crop_countH = (H//L)+2
    if(W <= L):
      crop_countW = 1
    else:
      crop_countW = (W//L)+2
    
    for k in range(0, crop_countH):
        if crop_countH > 1:
          img_k = crop(img, (0,W,(H-L)/(crop_countH-1)*k,L+(H-L)/(crop_countH-1)*k))
        elif crop_countH == 1:
          img_k = crop(img, (0,W,0,H))
    
        for j in range(0, crop_countW):
            if crop_countW > 1:
              img_t = crop(img_k, (0+((W-L)/(crop_countW-1)*j),L+((W-L)/(crop_countW-1)*j),0,L))
            elif crop_countW == 1:
              img_t = crop(img_k, (0, W, 0, H))
            if opt.format in ('png','tiff','jpg'):              
                img_save = rasterio.open(os.path.join(processed_dir, imgname+'_%d_%d'%(k,j) + '.' + opt.format),'w',driver='Gtiff',
                          width=output_w, height=output_h,
                          count=4,
                          crs=CRS,
                          transform=Transform,
                          dtype=dType
                          )
                img_save.write(img_t[:,:,0],3) #blue, rasterio color interpretation is RGBA
                img_save.write(img_t[:,:,1],2) #green
                img_save.write(img_t[:,:,2],1) #red 
                img_save.write(img_t[:,:,3],4) #nir
                img_save.close()

    t1 = time.time()
    print('running time:'+str(t1-t0))


