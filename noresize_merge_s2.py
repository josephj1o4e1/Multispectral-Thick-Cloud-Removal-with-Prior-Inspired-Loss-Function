import os
import sys
import argparse
import time
import rasterio
import cv2
import re
import numpy as np
from PIL import Image
from scipy.interpolate import CubicSpline

CURR_DIR = os.path.abspath(os.path.dirname(__file__)) #C:/.../Indoor
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="357p2p", help='')
parser.add_argument("--model_dir", type=str, default="", help='absolute path!!')
parser.add_argument('--datasetname', required=False,
  default='',  help='')
parser.add_argument('--L', required=False, type=int, default=512,  help='patchlength')
opt = parser.parse_args()

datasetname = ''
for i in range(len(re.split('/',opt.datasetname))):
  if i<len(re.split('/',opt.datasetname))-1:
    datasetname = datasetname + re.split('/',opt.datasetname)[i] + '_'
  else:
    datasetname = datasetname + re.split('/',opt.datasetname)[i]

modelname = opt.model + '_' + re.split('/|.pth',opt.model_dir)[-3] + '_' + re.split('/|.pth',opt.model_dir)[-2]

# ref_dir = os.path.join(CURR_DIR, 'dataset', opt.datasetname, 'Hazy')
ref_dir = os.path.join(CURR_DIR, 'dataset', opt.datasetname)

unprocessed_dir = os.path.join(CURR_DIR, 'image_results', 'dehazed_unmerged', datasetname, 'patch'+str(opt.L), modelname)
if not os.path.exists(unprocessed_dir):
  err='unprocessed_dir path: ' + unprocessed_dir + ' doesn\'t exist!'
  sys.exit(err)
# .../image_results/dehazed_unmerged/RawData_NTIRE2020_NH-HAZE_testHazy/patch1024_5_5/357pix2pixnoise_ssim_saved_models_generator_36

processed_dir = os.path.join(CURR_DIR, 'image_results', 'dehazed_merged', datasetname, 'patch'+str(opt.L), modelname) # .../image_results/resize_crop_shift/testNTIRE2020/patch1024_5_5
if not os.path.exists(processed_dir):
  os.makedirs(processed_dir, exist_ok=True)
# .../image_results/dehazed_merged/RawData_NTIRE2020_NH-HAZE_testHazy/patch1024_5_5/357pix2pixnoise_ssim_saved_models_generator_36

L = opt.L

def calc_ratio(patches):
    function = []
    for i in range(0,patches):
        start1=i*int(L/4)
        end1=i*int(L/4)+L
        
        X = np.linspace(start1, end1-1, num=20)
        X = np.delete(X, [5,6,7,8, 9,10, 11,12,13,14])
        x = np.arange(5) # 20/2
        y = np.power(x, 2)
        _y = np.flip(y)
        Y = np.concatenate((y, _y), axis=0)
        #------------------------------------------
        function.append(CubicSpline(X,Y))
        ratio = np.zeros(int((patches+3)/4*L))
        for C in range(start1,end1):
            ratio[C]+=function[i](C)

    return function, ratio


for full_img in os.listdir(ref_dir):
    all_index = full_img.split('.')[0]
    # Ref height, width
    ref_img = rasterio.open(os.path.join(ref_dir, full_img))
    H, W = ref_img.shape
    if(H <= L):
      crop_countH = 1
    else:
      crop_countH = (H//L)+2
    if(W <= L):
      crop_countW = 1
    else:
      crop_countW = (W//L)+2

    all_image=np.zeros((H,W,4,crop_countW))
    ratio_hor = np.zeros((H,W,4,crop_countW))
    t0 = time.time()
    print(all_index)
    to_combine = []

    for hor_index in range(0,crop_countW):
        image1=np.zeros((H,L,4,crop_countH))
        ratio_ver = np.zeros((H,L,4,crop_countH))
        for i in range(0,crop_countH):
            raster = rasterio.open(os.path.join(unprocessed_dir, str(all_index)+'_%s_%s.tiff'%(i ,hor_index)))
            band3 = raster.read(1) # R (fs2 is read as RGB-NIR from rasterio, opencv is read/write as BGR-NIR, we should flip the RBchannels)
            band2 = raster.read(2) # G
            band1 = raster.read(3) # B
            band4 = raster.read(4) # NIR
            img1 = np.dstack((band1,band2,band3,band4))

            start1=i*int((H-L)/(crop_countH-1))
            end1=i*int((H-L)/(crop_countH-1))+L

            X = np.linspace(start1, end1 - 1, num=20)
            X = np.delete(X, [5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
            x = np.arange(5) # 20/2
            y = np.power(x, 2)
            _y = np.flip(y)
            Y = np.concatenate((y, _y), axis=0)
            #------------------------------------------
            ratio = CubicSpline(X, Y)(np.arange(start1, end1))
            ratio = np.where(ratio <= 0, 0.0001, ratio).reshape(-1, 1, 1)
            ratio = np.tile(ratio, (1, opt.L, 4))
            # image1[start1:end1, :, :, i] = ratio * img1
            image1[start1:end1, :, :, i] = np.multiply(ratio, img1.astype(np.float64), out=img1.astype(np.float64), where=ratio!=0)
            ratio_ver[start1:end1, :, :, i] = ratio

        zz2=image1.sum(axis=3)
        ratio_ver = ratio_ver.sum(axis=3)
        zz2 = np.divide(zz2, ratio_ver, out=zz2, where=ratio_ver!=0)
        to_combine.append(zz2)

    for i, sub_img in enumerate(to_combine):
        #print(i)
        start1=i*int((W-L)/(crop_countW-1))
        end1=i*int((W-L)/(crop_countW-1))+L

        X = np.linspace(start1, end1 - 1, num=20)
        X = np.delete(X, [5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
        x = np.arange(5) # 20/2
        y = np.power(x, 2)
        _y = np.flip(y)
        Y = np.concatenate((y, _y), axis=0)
        #------------------------------------------
        ratio = CubicSpline(X, Y)(np.arange(start1, end1))
        ratio = np.where(ratio <= 0.0001, 1, ratio).reshape(1, -1, 1)
        ratio = np.tile(ratio, (H, 1, 4))
        # all_image[:, start1:end1, :, i] = ratio * sub_img
        all_image[:, start1:end1, :, i] = np.multiply(ratio, sub_img.astype(np.float64), out=sub_img.astype(np.float64), where=ratio!=0)
        ratio_hor[:, start1:end1, :, i] = ratio

    zz3 = all_image.sum(axis=3)
    ratio_hor = ratio_hor.sum(axis=3)
    zz3 = np.divide(zz3, ratio_hor, out=zz3, where=ratio_hor!=0)

    cv2.imwrite(os.path.join(processed_dir, str(all_index)+'.tiff'), zz3.astype(np.uint16))

    print('Done!')
    t1 = time.time()
    print('running time:'+str(t1-t0))
