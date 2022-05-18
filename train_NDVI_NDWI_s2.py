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
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

# from datasets2 import *
from datasets import *
from model import *

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch
from skimage.measure import compare_psnr, compare_ssim
import statistics
from pytorch_msssim import ms_ssim, MS_SSIM

import torch.backends.cudnn as cudnn
cudnn.benchmark = True
cudnn.fastest = True

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=80, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="DS5_2020_only", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=8, help="size of the batches")
parser.add_argument("--g_lr", type=float, default=0.0001, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=512, help="size of image height")
parser.add_argument("--img_width", type=int, default=512, help="size of image width")
parser.add_argument("--channels", type=int, default=4, help="number of image channels")
parser.add_argument("--pixval_max", type=int, default=10000, help="sentinel-2 is 32767, we now clamp it to (0, 10000)")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between saving generator samples")
parser.add_argument("--checkpoint_interval", type=int, default=2, help="interval between model checkpoints")
parser.add_argument("--model", type=str, default="NTIRE2018_Multiscale_NDVI_NDWI", help="the model's name")
parser.add_argument('--gpus', nargs='*', type=int, default=[0], help='To use the DataParallel method')
parser.add_argument("--gpuid", type=int, default=0, help="record which gpu it is used")
parser.add_argument("--pretrained_model_pathG", type=str, default="", help="the pretrainedmodel's path. Ex: saved_models/NTIRE2018_Multiscale/DS5_2020/generator_20.pth")
# parser.add_argument("--init_ortho", type=str, default="False", help="True or False")
opt = parser.parse_args()
print(opt)

from torch.utils.tensorboard import SummaryWriter       
if opt.pretrained_model_pathG != "":
    writer = SummaryWriter(comment="_gpu"+str(opt.gpuid)+"_"+opt.model+"_"+opt.dataset_name+"_G-"+re.split('/|.pth',opt.pretrained_model_pathG)[-2])
else:
    writer = SummaryWriter(comment="_gpu"+str(opt.gpuid)+"_"+opt.model+"_"+opt.dataset_name)

current_time = datetime.datetime.now().strftime('%b%d_%H-%M-%S')

cuda = True if torch.cuda.is_available() else False
print("cuda!!!!")
print(cuda)

# Initialize generator
if opt.model=="NTIRE2018_Multiscale_NDVI_NDWI" :
    generator = Satellite4in4out_NDVI_NDWI_Dense_rain_cvprw3()
    opt.channels = 4
    opt.model = "Satellite4in4out_" + opt.model
else:
    sys.exit("--model should be: NTIRE2018_Multiscale_NDVI_NDWI")

# generator.apply(weights_init)
os.makedirs("images/%s/%s/%s" % (opt.model, opt.dataset_name, current_time), exist_ok=True)
os.makedirs("saved_models_mgpu/%s/%s/%s" % (opt.model, opt.dataset_name, current_time), exist_ok=True)
os.makedirs("saved_models_val_mgpu/%s/%s/%s" % (opt.model, opt.dataset_name, current_time), exist_ok=True)

opt.img_width=opt.img_height
input_shape = (opt.channels, opt.img_height, opt.img_width)

# Loss functions
# mae_loss = torch.nn.L1Loss()
mse_loss = nn.MSELoss()
ms_ssim_module = MS_SSIM(data_range=1, size_average=True, channel=opt.channels) # size_average=True for scalar output
# ms_ssim_module = MS_SSIM(data_range=255, size_average=True, channel=4) # MS_SSIM is torch.nn
# ms_ssim_module = MS_SSIM(data_range=255, size_average=True, channel=3) # MS_SSIM is torch.nn
def ms_ssim_loss(X, Y):
    return 1 - ms_ssim_module(X,Y)


# ##################################for pytorch0.3 models
# print("Before modify!!\n")
# # print("Pretrained-model:\n")
# if opt.pretrained_model_pathG != '':
#   checkpoint = torch.load(opt.pretrained_model_pathG)
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

device = torch.device("cuda")
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(n) for n in opt.gpus])

if opt.pretrained_model_pathG != "":
    generator.load_state_dict(torch.load(opt.pretrained_model_pathG))

if len(opt.gpus) > 1:
    generator = nn.DataParallel(generator, device_ids=opt.gpus)
    
if cuda:
    generator = generator.cuda()
    


# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.g_lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

train_dataset = SatelliteImageDataset("dataset/%s/train%s" % (opt.dataset_name, str(opt.img_height)), input_shape, mode="train", pixval_max=opt.pixval_max)
val_dataset = SatelliteImageDataset("dataset/%s/val%s" % (opt.dataset_name, str(opt.img_height)), input_shape, mode="val", pixval_max=opt.pixval_max)

#train_sampler = torch.utils.data.distributed.DistributedSampler(
#    train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
#val_sampler = torch.utils.data.distributed.DistributedSampler(
#    val_dataset, num_replicas=hvd.size(), rank=hvd.rank())

dataloader = DataLoader(
    train_dataset,
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu,
)
val_dataloader = DataLoader(
    val_dataset,
    # batch_size=opt.batch_size,
    batch_size=1,
    shuffle=False,
    num_workers=1,
)

# def norm_ip(img, min, max):
#     img.clamp_(min=min, max=max)
#     img.add_(-min).div_(max - min)

#     return img

# def norm_range(t, range):
#     if range is not None:
#         norm_ip(t, range[0], range[1])
#     else:
#         norm_ip(t, t.min(), t.max())
#     return norm_ip(t, t.min(), t.max())

def save_image(tensor, imgpath):
    raster = rasterio.open('/home/sychuang0909/dehazing_nas/Datasets/Sentinel-2/20210303/val/9/hazy/T31SCD_20201014T104031_B08_10m.jp2')
    CRS=raster.crs
    Transform=raster.transform
    dType=raster.dtypes[0]
    tensor = torch.squeeze(tensor)
    tensor = (tensor*0.5)+0.5
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

def psnr_ssim_calc(tensor_fB, tensor_rB):    
    tensor_fB = torch.squeeze(tensor_fB)
    tensor_rB = torch.squeeze(tensor_rB)        
    tensor_fB, tensor_rB = (tensor_fB*0.5)+0.5, (tensor_rB*0.5)+0.5

    ndarr_fB = tensor_fB.mul(opt.pixval_max).clamp_(0, opt.pixval_max).permute(1, 2, 0).to('cpu').detach().numpy()
    ndarr_fB = ndarr_fB.astype(np.uint16)

    ndarr_rB = tensor_rB.mul(opt.pixval_max).clamp_(0, opt.pixval_max).permute(1, 2, 0).to('cpu').detach().numpy()
    ndarr_rB = ndarr_rB.astype(np.uint16)
    
    psnr = compare_psnr(ndarr_fB, ndarr_rB)
    ssim = compare_ssim(ndarr_fB, ndarr_rB, multichannel = True)

    return psnr, ssim
      
    
def sample_images(batches_done, psnr_ssim=False):
    """Saves a generated sample from the validation set"""    
    total_loss = 0.0
    if psnr_ssim:
        psnr_list = []
        ssim_list = []
    generator.eval()
    for batch_id, imgs in enumerate(val_dataloader):
        real_A, real_B = Variable(imgs["A"].type(Tensor)), Variable(imgs["B"].type(Tensor))
        filename = imgs["name"]
        fake_B, fake_B2 = generator(real_A)
        total_loss += mse_loss(fake_B, real_B).item() * real_A.size(0)

        for fB, rB, filename_A in zip(fake_B, real_B, filename):
            if psnr_ssim:
                if batch_id < 40:
                    os.makedirs("images/%s/%s/%s/epoch%d_dh" % (opt.model, opt.dataset_name, current_time, epoch), exist_ok=True)
                    os.makedirs("images/%s/%s/%s/epoch%d_gt" % (opt.model, opt.dataset_name, current_time, epoch), exist_ok=True)
                    filename_A_fB = filename_A.split('_')[0]+'_dh.tiff'
                    filename_A_rB = filename_A.split('_')[0]+'_gt.tiff'
                    valimg_path_fB = os.path.join("images",opt.model,opt.dataset_name,current_time,"epoch"+str(epoch)+'_dh',filename_A_fB)
                    valimg_path_rB = os.path.join("images",opt.model,opt.dataset_name,current_time,"epoch"+str(epoch)+'_gt',filename_A_rB)
                    save_image(fB.cpu().detach(), valimg_path_fB)
                    save_image(rB.cpu().detach(), valimg_path_rB)
                                
                psnr, ssim = psnr_ssim_calc(fB.cpu().detach(), rB.cpu().detach()) # will automatically convert data_range to the data_type
                psnr_list.append(psnr)
                ssim_list.append(ssim)

    val_loss = total_loss/len(val_dataset)
    generator.train()
    if psnr_ssim:
        return (val_loss, (statistics.mean(psnr_list), statistics.mean(ssim_list)))
        # return (val_loss, statistics.mean(psnr_list))
    return val_loss

# ----------
#  Training
# ----------
# outfilename=os.path.join(CURR_DIR,"tensortensor.txt")
# sys.stdout = open(outfilename, "w")
min_val = 0.05
min_epoch = opt.epoch

prev_time = time.time()
running_loss = 0.0
running_loss_withNDVINDWI = 0.0
running_totalloss = 0.0
for epoch in range(opt.epoch, opt.n_epochs):
    print("\nEpoch   %d/%d" %(epoch, opt.n_epochs))
    print("-----------------------------------")
    for i, batch in enumerate(dataloader):
        
        # Set model input
        real_A = Variable(batch["A"].type(Tensor))
        real_B = Variable(batch["B"].type(Tensor))
        real_B2 = Variable(batch["B2"].type(Tensor))
        
        # -------------------------------
        #  Train Generator
        # -------------------------------

        optimizer_G.zero_grad()
        fake_B, fake_B2 = generator(real_A) # fake_B2 is for NDVI/NDWI calculation
        
        # Pixelwise loss
        loss_pixel_LR = mse_loss(fake_B, real_B) # added pixelwise mse loss

        # added pixelwise NDVI/NDWI loss
        fake_B_R, fake_B_NIR = (fake_B2[:,0,:,:]+1)/2, (fake_B2[:,3,:,:]+1)/2 # denorm to (0, 1)
        fake_B_ndvi = (fake_B_NIR - fake_B_R) / (fake_B_NIR + fake_B_R) # (8,1,512,512) 
        fake_B_ndvi2 = fake_B_ndvi.detach().clone()
        fake_B_ndvi2[fake_B_ndvi2!=fake_B_ndvi2]=0
        fake_B_ndvi[fake_B_ndvi!=fake_B_ndvi]=torch.mean(fake_B_ndvi2) # if nan due to 0/0, replace mean
        real_B_R, real_B_NIR = (real_B2[:,0,:,:]+1)/2, (real_B2[:,3,:,:]+1)/2 # denorm to (0, 1)
        real_B_ndvi = (real_B_NIR - real_B_R) / (real_B_NIR + real_B_R)
        real_B_ndvi2 = real_B_ndvi.detach().clone()
        real_B_ndvi2[real_B_ndvi2!=real_B_ndvi2]=0
        real_B_ndvi[real_B_ndvi!=real_B_ndvi]=torch.mean(real_B_ndvi2)
        loss_pixel_NDVI = mse_loss(fake_B_ndvi, real_B_ndvi)

        fake_B_G = (fake_B2[:,1,:,:]+1)/2 # denorm to (0, 1)
        fake_B_ndwi = (fake_B_G - fake_B_NIR) / (fake_B_G + fake_B_NIR)
        fake_B_ndwi2 = fake_B_ndwi.detach().clone()
        fake_B_ndwi2[fake_B_ndwi2!=fake_B_ndwi2]=0
        fake_B_ndwi[fake_B_ndwi!=fake_B_ndwi]=torch.mean(fake_B_ndwi2)
        real_B_G = (real_B2[:,1,:,:]+1)/2 # denorm to (0, 1)
        real_B_ndwi = (real_B_G - real_B_NIR) / (real_B_G + real_B_NIR)
        real_B_ndwi2 = real_B_ndwi.detach().clone()
        real_B_ndwi2[real_B_ndwi2!=real_B_ndwi2]=0
        real_B_ndwi[real_B_ndwi!=real_B_ndwi]=torch.mean(real_B_ndwi2)
        loss_pixel_NDWI = mse_loss(fake_B_ndwi, real_B_ndwi)

        # added pixelwise ssim loss
        # needed to denormalize from (-1, 1) to (0, 1) to calculate ms-ssim in pytorch
        fake_B_denorm=(fake_B+1)/2
        real_B_denorm=(real_B+1)/2
        loss_pixel_LRssim = ms_ssim_loss(fake_B_denorm, real_B_denorm)
        running_loss_withNDVINDWI += (0.01*loss_pixel_NDVI.item() + 0.01*loss_pixel_NDWI.item())
        running_loss += loss_pixel_LR.item()

        # ----------------------------------
        # Total Loss (Generator)
        # ----------------------------------

        loss_G = loss_pixel_LR + loss_pixel_LRssim + 0.01*loss_pixel_NDVI + 0.01*loss_pixel_NDWI
        running_totalloss += loss_G.item()

        loss_G.backward()
        optimizer_G.step()

        # --------------
        #  Log Progress
        # --------------
                
        # Determine approximate time left
        batches_done = epoch * len(dataloader) + i
        batches_left = opt.n_epochs * len(dataloader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

        if batches_done % opt.sample_interval == 0 and batches_done != 0:
            val_score = sample_images(batches_done)
            train_score = running_loss/opt.sample_interval
            writer.add_scalar('Train/train_score',train_score,batches_done)
            writer.add_scalar('Val/val_score',val_score,batches_done)
            writer.add_scalar('Train/loss_pixel_LR',loss_pixel_LR.item(),batches_done)
            writer.add_scalar('Train/loss_pixel_LRssim',loss_pixel_LRssim.item(),batches_done)
            writer.add_scalar('Train/train_score_with_NDVI_NDWI',(running_loss_withNDVINDWI/opt.sample_interval),batches_done)
            writer.add_scalar('Train/total_loss',(running_totalloss/opt.sample_interval),batches_done)
            writer.flush() # if too slow then flush every epoch. now, it is every opt.sample_interval batches
            print("\n******************************************************************")
            print("#%d batch Val stage  val MSE Loss: %.5f  train Loss: %.5f" %(batches_done, val_score, train_score))
            print("******************************************************************\n")
            running_loss = 0.0
            running_loss_withNDVINDWI = 0.0
            running_totalloss = 0.0
            if epoch >= 5 and val_score < min_val:
                torch.save(generator.state_dict(), "saved_models_val_mgpu/%s/%s/%s/generator_%d.pth" % (opt.model, opt.dataset_name, current_time, epoch)) # .module
                min_val = val_score
                min_epoch = epoch

        # Print log
        sys.stdout.write(
                "\n[Epoch %d/%d] [Batch %d/%d] [pixel_LR: %f, pixel_LRssim: %f, NDVIloss: %f, NDWIloss: %f, min_val: %f] ETA: %s"
            % (
                epoch,
                opt.n_epochs,
                i,
                len(dataloader),
                loss_pixel_LR.item(),
                loss_pixel_LRssim.item(),
                loss_pixel_NDVI.item(),
                loss_pixel_NDWI.item(),
                min_val,
                time_left,
            )
        )

    if (opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0) or epoch == opt.n_epochs-1:
        # compute psnr and ssim and validate
        batches_done = (epoch+1) * len(dataloader)
        val_loss, psnr_ssim = sample_images(batches_done, True)
        print("\n*************************************************")
        print("ckpt sample img and validation")
        print("%d Epoch val L2 loss: %f" %(epoch, val_loss))
        print("%d Epoch val psnr: %f   ssim: %f" %(epoch, psnr_ssim[0], psnr_ssim[1]))
        # print("%d Epoch val psnr: %f" %(epoch, psnr_ssim))
        print("%d Epoch min_val: %.5f at %d epoch" %(epoch, min_val, min_epoch))
        print("*************************************************\n\n")
        writer.add_scalar('Val/psnr',psnr_ssim[0],epoch)
        writer.add_scalar('Val/ssim',psnr_ssim[1],epoch)
        # writer.add_scalar('Val/psnr',psnr_ssim,epoch)
        writer.flush()        
        # Save model checkpoints
        torch.save(generator.state_dict(), "saved_models_mgpu/%s/%s/%s/generator_%d.pth" % (opt.model, opt.dataset_name, current_time, epoch)) # .module

         
# sys.stdout.close()
    
    

