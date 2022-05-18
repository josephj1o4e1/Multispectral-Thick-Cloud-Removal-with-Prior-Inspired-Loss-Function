import argparse
import os
import numpy as np
import math
import itertools
import datetime
import time
import sys
import re
import cv2
import rasterio

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from datasets import *
from model import *

import torch.nn as nn
import torch.nn.functional as F
import torch
from skimage.measure import compare_psnr, compare_ssim, compare_nrmse
import statistics

# CUDA_VISIBLE_DEVICES=0 python3 test_NDVI_NDWI_s2.py --model_dir /home/sychuang0909/sychuang0909_4journal/Code/saved_models_mgpu/Satellite4in4out_NTIRE2018_Multiscale_NDVI_NDWI/20210303_augmented/Mar12_12-12-23/generator_28.pth --model NTIRE2018_Multiscale_NDVI_NDWI --datasetname 20210303/val --L 512

CURR_DIR = os.path.abspath(os.path.dirname(__file__))
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=8, help="number of latent codes")
parser.add_argument("--model", type=str, default="NTIRE2018_Multiscale_NDVI_NDWI", help='')
parser.add_argument("--model_dir", type=str, default="", help='absolute path!!')
parser.add_argument("--datasetname", type=str, default="RawData/NTIRE2020/NH-HAZE_testHazy")
parser.add_argument("--L", type=int, default=512)
parser.add_argument('--channels', required=False, type=int,  default=4,  help='')
parser.add_argument("--pixval_max", type=int, default=10000, help="sentinel-2 is 32767")
opt = parser.parse_args()
print(opt)

cuda = True if torch.cuda.is_available() else False
print(cuda)

# Initialize generator
if opt.model=="NTIRE2018_Multiscale_NDVI_NDWI" :
    generator = Satellite4in4out_NDVI_NDWI_Dense_rain_cvprw3()
    opt.channels = 4
    opt.model = "Satellite4in4out_" + opt.model
else:
    sys.exit("--model should be: NTIRE2018_Multiscale_NDVI_NDWI")


if cuda:
    generator = generator.cuda()

generator.load_state_dict(torch.load(opt.model_dir)) # model

# ##################################for pytorch0.3 models
# print("Before modify!!\n")
# # print("Pretrained-model:\n")
# if opt.model_dir != '':
#   checkpoint = torch.load(opt.model_dir)
#   pretrained_state_dict = checkpoint['model_state_dict']
#   print("Start modify!!\n")
#   new_state_dict = OrderedDict()
#   for i, (key, val) in enumerate(pretrained_state_dict.items()):
#     x = re.findall("dense_block[0-9]+\.denselayer[0-9]+\.(?:norm\.|conv\.)[0-9]+\.\S+", key) # x is list of string
#     if x :
#       normx = re.findall("norm\.[0-9]+",x[0]) # normx[0] == 'norm.1' or 'norm.2' start index
#       convx = re.findall("conv\.[0-9]+",x[0]) # convx[0] == 'conv.1' or 'conv.2' start index
#       if normx :
#           result_key = re.sub('norm\.[0-9]+','norm'+normx[0][5],x[0])
#       elif convx :
#           result_key = re.sub('conv\.[0-9]+','conv'+convx[0][5],x[0])
#       new_state_dict[result_key] = val
#     else :
#       new_state_dict[key] = val
  
#   generator.load_state_dict(new_state_dict)
# ##################################

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
opt.pixval_max=32767

datasetname = ''
for i in range(len(re.split('/',opt.datasetname))):
  if i<len(re.split('/',opt.datasetname))-1:
    datasetname = datasetname + re.split('/',opt.datasetname)[i] + '_'
  else:
    datasetname = datasetname + re.split('/',opt.datasetname)[i]

modelname = opt.model + '_' + re.split('/|.pth',opt.model_dir)[-3] + '_' + re.split('/|.pth',opt.model_dir)[-2] 

# test_dataset = Satellitehidden_test_dataset(os.path.join(CURR_DIR, 'image_results', 'noresize_crop_shift', datasetname, 'patch'+str(opt.L)), opt.channels)

# test_dataloader = DataLoader(
#     test_dataset,
#     batch_size=opt.batch_size,
#     shuffle=False,
#     num_workers=1,
# )

def norm_ip(img, min, max):
    img.clamp_(min=min, max=max)
    img.add_(-min).div_(max - min)

    return img

def norm_range(t, range):
    if range is not None:
        norm_ip(t, range[0], range[1])
    else:
        norm_ip(t, t.min(), t.max())
    return norm_ip(t, t.min(), t.max())

def save_image(tensor, imgpath):
    # for convenience, val_batchsize=1, don't need grid
    # grid = make_grid(tensor, nrow=nrow, padding=padding, pad_value=pad_value,
    #                  normalize=normalize, range=range, scale_each=scale_each)
    raster = rasterio.open('/home/sychuang0909/dehazing_nas/Datasets/Sentinel-2/20210303/val/9/hazy/T31SCD_20201014T104031_B08_10m.jp2')
    CRS=raster.crs
    Transform=raster.transform
    dType=raster.dtypes[0]
    tensor = torch.squeeze(tensor)
    tensor = norm_range(tensor, None)
    # ndarr = tensor.mul(opt.pixval_max).add_(0.5).clamp_(0, opt.pixval_max).permute(1, 2, 0).to('cpu').detach().numpy()
    ndarr = tensor.mul(opt.pixval_max).clamp_(0, opt.pixval_max).permute(1, 2, 0).to('cpu').detach().numpy()
    ndarr = ndarr.astype(np.uint16)
    img_save = rasterio.open(imgpath,'w',driver='Gtiff',
            width=ndarr.shape[1], height=ndarr.shape[0],
            count=4,
            crs=CRS,
            transform=Transform,
            dtype=dType
            )
    img_save.write(ndarr[:,:,0],1) #blue, rasterio color interpretation is RGBA
    img_save.write(ndarr[:,:,1],2) #green
    img_save.write(ndarr[:,:,2],3) #red 
    img_save.write(ndarr[:,:,3],4) #nir
    img_save.close()
    # tensor = torch.squeeze(tensor)
    # tensor = norm_range(tensor, None)
    # # ndarr = tensor.mul(opt.pixval_max).add_(0.5).clamp_(0, opt.pixval_max).permute(1, 2, 0).to('cpu').detach().numpy()
    # ndarr = tensor.mul(opt.pixval_max).clamp_(0, opt.pixval_max).permute(1, 2, 0).to('cpu').detach().numpy()
    # ndarr = ndarr.astype(np.uint16)
    # if opt.channels==4:
    #     ndarr = np.dstack((ndarr[...,2],ndarr[...,1],ndarr[...,0],ndarr[...,3])) # cv2...needs to be converted to BGR before saving, NIR doesn't change its place
    # elif opt.channels==3:
    #     ndarr = np.dstack((ndarr[...,2],ndarr[...,1],ndarr[...,0]))
    # else:
    #     sys.exit("opt.channels==3 or 4")
    
    # cv2.imwrite(imgpath, ndarr.astype(np.uint16))

# test_dataset = Satellitehidden_test_dataset(os.path.join(CURR_DIR, 'image_results', 'noresize_crop_shift', datasetname, 'patch'+str(opt.L)), opt.channels, opt.pixval_max)

# test_dataloader = DataLoader(
#     test_dataset,
#     batch_size=opt.batch_size,
#     shuffle=False,
#     num_workers=1,
# )

# generator.eval()
# with torch.no_grad():    
#     for batch_id, imgs in enumerate(test_dataloader):
#         t0=time.time()
#         real_A = Variable(imgs["A"].type(Tensor))
#         filename = imgs["name"]
#         fake_B, fake_B2 = generator(real_A)
#         fake_B = (fake_B*0.5)+0.5
        
#         for fB, filename_A in zip(fake_B, filename):
#             dest_dir = os.path.join(CURR_DIR, 'image_results', 'dehazed_unmerged', datasetname, 'patch'+str(opt.L), modelname)
#             os.makedirs(dest_dir, exist_ok=True)
#             save_image(fB,os.path.join(dest_dir, filename_A)) 
#         t1=time.time()
#         print('run time per imagepatch='+str(t1-t0))


#################################################################
# test_dataset = Satellitehidden_test_dataset_TMP(os.path.join(CURR_DIR, 'image_results', 'noresize_crop_shift', datasetname, 'patch'+str(opt.L)), os.path.join(CURR_DIR, 'image_results', 'noresize_crop_shift', datasetname+'gt', 'patch'+str(opt.L)), opt.channels, opt.pixval_max)
# test_dataset = Satellitehidden_test_dataset_TMP(os.path.join(CURR_DIR, 'image_results', 'resizeonly', datasetname, 'patch'+str(opt.L)), os.path.join(CURR_DIR, 'image_results', 'resizeonly', datasetname+'gt', 'patch'+str(opt.L)), opt.channels, opt.pixval_max)
test_dataset = Satellitehidden_test_dataset_TMP("dataset/smalltest/val512/Hazy", "dataset/smalltest/val512/GT", opt.channels, opt.pixval_max)
test_dataloader = DataLoader(
    test_dataset,
    batch_size=opt.batch_size,
    shuffle=False,
    num_workers=1,
)



# psnr_list=[]
# generator.eval()
# with torch.no_grad():    
#     for batch_id, imgs in enumerate(test_dataloader):
#         t0=time.time()
#         real_A = Variable(imgs["A"].type(Tensor))
#         real_B = Variable(imgs["B"].type(Tensor))
#         filename = imgs["name"]
#         fake_B, fake_B2 = generator(real_A)
#         fake_B, real_B = (fake_B*0.5)+0.5, (real_B*0.5)+0.5
        
#         # for fB, filename_A in zip(fake_B, filename):
#         for fB, rB, filename_A in zip(fake_B, real_B, filename):
#             fake_B_ndarray = fB.cpu().detach().numpy() # tensor 2 ndarray ?
#             real_B_ndarray = rB.cpu().detach().numpy() # tensor 2 ndarray ?
#             psnr = compare_psnr(real_B_ndarray, fake_B_ndarray)
#             psnr_list.append(psnr)
#             dest_dir = os.path.join(CURR_DIR, 'image_results', 'dehazed_unmerged', datasetname, 'patch'+str(opt.L), modelname)
#             os.makedirs(dest_dir, exist_ok=True)
#             save_image(fB,os.path.join(dest_dir, filename_A))            
#         t1=time.time()
#         print('run time per imagepatch='+str(t1-t0))
#     print("MEAN PSNR: ")
#     print(statistics.mean(psnr_list))

psnr_list=[]
generator.eval()
with torch.no_grad():    
    for batch_id, imgs in enumerate(test_dataloader):
        t0=time.time()
        real_A = Variable(imgs["A"].type(Tensor))
        real_B = Variable(imgs["B"].type(Tensor))
        filenameA = imgs["nameA"]
        filenameB = imgs["nameB"]
        fake_B, fake_B2 = generator(real_A)
        fake_B, real_B = (fake_B*0.5)+0.5, (real_B*0.5)+0.5
        
        for fB, rB, rA, filename_A, filename_B in zip(fake_B, real_B, real_A, filenameA, filenameB):
            fake_B_ndarray = fB.cpu().detach().numpy() # tensor 2 ndarray ?
            real_B_ndarray = rB.cpu().detach().numpy() # tensor 2 ndarray ?
            psnr = compare_psnr(real_B_ndarray, fake_B_ndarray)
            psnr_list.append(psnr)
            print("psnr:")
            print(psnr)
            dest_dir_dh = os.path.join(CURR_DIR, 'dummy', datasetname+'dh', 'patch'+str(opt.L), modelname)
            dest_dir_gt = os.path.join(CURR_DIR, 'dummy', datasetname+'gt', 'patch'+str(opt.L), modelname)
            dest_dir_hz = os.path.join(CURR_DIR, 'dummy', datasetname, 'patch'+str(opt.L), modelname)
            os.makedirs(dest_dir_dh, exist_ok=True)
            os.makedirs(dest_dir_gt, exist_ok=True)
            os.makedirs(dest_dir_hz, exist_ok=True)
            save_image(fB,os.path.join(dest_dir_dh, filename_A))
            save_image(rB,os.path.join(dest_dir_gt, filename_B))
            save_image(rA,os.path.join(dest_dir_hz, filename_A))
        t1=time.time()
        print('run time per imagepatch='+str(t1-t0))
    print("MEAN PSNR: ")
    print(statistics.mean(psnr_list))

