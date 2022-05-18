import glob
import random
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageFile
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from math import ceil
import csv
import tifffile as tiff
import rasterio
from rasterio.transform import Affine
import sys

def read(file):
    with rasterio.open(file) as src:
        band1=src.read(1)
        band2=src.read(2)
        band3=src.read(3)
        band4=src.read(4)
        raster=np.dstack((band1,band2,band3,band4))
        return raster, src.crs, src.transform, src.dtypes[0]

class ImageDataset(Dataset):
    def __init__(self, root, input_shape, mode="train"):
        self.transform = transforms.Compose(
            [
                transforms.Resize(input_shape[-2:]),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

        self.shape = input_shape[-2:]
        self.files_A = sorted(glob.glob(os.path.join(root,"Hazy") + "/*.*"))
        self.files_B = sorted(glob.glob(os.path.join(root,"GT") + "/*.*"))
        self.mode = mode

    def __getitem__(self, index):

        ImageFile.LOAD_TRUNCATED_IMAGES = True
        img_A = Image.open(self.files_A[index % len(self.files_A)])
        img_B = Image.open(self.files_B[index % len(self.files_B)])

        path_A = self.files_A[index % len(self.files_A)]
        filename_A = os.path.basename(path_A)
        if self.mode == 'test':
            width = int(ceil(img_A.size[0]/640))
            height = int(ceil(img_A.size[1]/640))
            img_A_list = []

            coordinate = [0,384]
            for i in range(width*height):
                left = coordinate[int(i%width)]
                up = coordinate[int(i//width)]
                crop_A = img_A.crop((left, up , left+640, up+640))
                crop_A = self.transform(crop_A)
                img_A_list.append(crop_A)
            img = torch.stack(img_A_list, dim=0)
            return {"A": img, "name": filename_A, "width": width, "height": height}

        if self.mode == 'train':
            #random resize crop
            #i, j, h, w = transforms.RandomResizedCrop.get_params(img_A, (0.8,1.0), (0.75,1.3333333333))
            #img_A = TF.crop(img_A, i, j, h, w)
            #img_B = TF.crop(img_B, i, j, h, w)
            #flip randomly
            if np.random.random() < 0.5:
                img_A = Image.fromarray(np.array(img_A)[:, ::-1, :], "RGB")
                img_B = Image.fromarray(np.array(img_B)[:, ::-1, :], "RGB")

        img_A = self.transform(img_A)
        img_B = self.transform(img_B)

        return {"A": img_A, "B": img_B, "name": filename_A}

    def __len__(self):
        return len(self.files_A)

class SatelliteImageDataset(Dataset):
    def __init__(self, root, input_shape, mode="train", pixval_max=10000):
        self.transform = transforms.Compose(
            [
                # transforms.Resize(input_shape[-2:]), # Actually the Data are all cropped before so this line is useless
                # transforms.ToTensor(),
                # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

        self.shape = input_shape[-2:]
        self.channels = input_shape[-3]
        self.files_A = sorted(glob.glob(os.path.join(root,"Hazy") + "/*.*"))
        self.files_B = sorted(glob.glob(os.path.join(root,"GT") + "/*.*"))
        self.mode = mode
        self.pixval_max = pixval_max/1.0

    def __getitem__(self, index):

        # ImageFile.LOAD_TRUNCATED_IMAGES = True
        # img_A = Image.open(self.files_A[index % len(self.files_A)])
        # img_B = Image.open(self.files_B[index % len(self.files_B)])
        img_A = tiff.imread(self.files_A[index % len(self.files_A)]) # 4 channel ndarray
        img_B = tiff.imread(self.files_B[index % len(self.files_B)])
        if self.channels==3:
            img_A = img_A[:,:,0:3]
            img_B = img_B[:,:,0:3]

        path_A = self.files_A[index % len(self.files_A)]
        filename_A = os.path.basename(path_A)
        if self.mode == 'test':
            width = int(ceil(img_A.size[0]/640))
            height = int(ceil(img_A.size[1]/640))
            img_A_list = []

            coordinate = [0,384]
            for i in range(width*height):
                left = coordinate[int(i%width)]
                up = coordinate[int(i//width)]
                crop_A = img_A.crop((left, up , left+640, up+640))
                crop_A = self.transform(crop_A)
                img_A_list.append(crop_A)
            img = torch.stack(img_A_list, dim=0)
            return {"A": img, "name": filename_A, "width": width, "height": height}

        if self.mode == 'train':
            #random resize crop
            #i, j, h, w = transforms.RandomResizedCrop.get_params(img_A, (0.8,1.0), (0.75,1.3333333333))
            #img_A = TF.crop(img_A, i, j, h, w)
            #img_B = TF.crop(img_B, i, j, h, w)
            #flip randomly
            if np.random.random() < 0.5:
                # img_A = Image.fromarray(np.array(img_A)[:, ::-1, :], "RGB")
                # img_B = Image.fromarray(np.array(img_B)[:, ::-1, :], "RGB")
                img_A = img_A[:, ::-1, :]
                img_B = img_B[:, ::-1, :]

        # img_A = self.transform(img_A)
        # img_B = self.transform(img_B)

        # ----------------------------

        # TRANSFORM!!

        # ToTensor() for uint16
        img_A = img_A/1.0 # torch.from_numpy: The only supported types are: float64, float32, float16, int64, int32, int16, int8, and uint8. uint16/1.0 -> float16
        img_A2 = torch.from_numpy((img_A.transpose((2, 0, 1)).copy())) # contiguous?
        img_A2 = img_A2.clamp_(0, self.pixval_max)
        img_A2 = img_A2.float()/self.pixval_max # why can't I use img_A2.float().div_(10000)??????

        img_B = img_B/1.0 # torch.from_numpy: The only supported types are: float64, float32, float16, int64, int32, int16, int8, and uint8. uint16/1.0 -> float16
        img_B2 = torch.from_numpy((img_B.transpose((2, 0, 1)).copy())) # contiguous?     
        img_B2 = img_B2.clamp_(0, self.pixval_max)
        img_B2 = img_B2.float()/self.pixval_max
    
        # Normalize() for float16
        # mean = torch.as_tensor([0.5, 0.5, 0.5, 0.5], dtype=img_A2.dtype, device=img_A2.device)
        # std = torch.as_tensor([0.5, 0.5, 0.5, 0.5], dtype=img_A2.dtype, device=img_A2.device)
        mean = torch.as_tensor(np.full((self.channels,), 0.5), dtype=img_A2.dtype, device=img_A2.device)
        std = torch.as_tensor(np.full((self.channels,), 0.5), dtype=img_A2.dtype, device=img_A2.device)
        if mean.ndim == 1:
            mean = mean.view(-1, 1, 1)
        if std.ndim == 1:
            std = std.view(-1, 1, 1)
        
        img_A2.sub_(mean).div_(std)      
        img_B22 = img_B2 # without normalization
        img_B2.sub_(mean).div_(std)
    
        # ----------------------------

        return {"A": img_A2, "B": img_B2, "B2": img_B22, "name": filename_A}

    def __len__(self):
        return len(self.files_A)



class hidden_test_dataset(Dataset):
    def __init__(self, root):
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        self.files_A = sorted(glob.glob(root + "/*.*"))

    def __getitem__(self, index):

        ImageFile.LOAD_TRUNCATED_IMAGES = True
        img_A = Image.open(self.files_A[index % len(self.files_A)])

        path_A = self.files_A[index % len(self.files_A)]
        filename_A = os.path.basename(path_A)

        img_A = self.transform(img_A)

        return {"A": img_A, "name": filename_A}

    def __len__(self):
        return len(self.files_A)

class Satellitehidden_test_dataset(Dataset):
    def __init__(self, root, channels, pixval_max=10000):
        self.transform = transforms.Compose(
            [
                # transforms.ToTensor(),
                # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        self.files_A = sorted(glob.glob(root + "/*.*"))
        self.channels = channels
        self.pixval_max = pixval_max/1.0

    def __getitem__(self, index):

        # ImageFile.LOAD_TRUNCATED_IMAGES = True
        # img_A = Image.open(self.files_A[index % len(self.files_A)])
        img_A = tiff.imread(self.files_A[index % len(self.files_A)])
        if self.channels==3:
            img_A = img_A[:,:,0:3]

        path_A = self.files_A[index % len(self.files_A)]
        filename_A = os.path.basename(path_A) 

        # TRANSFORM!
        # img_A = self.transform(img_A)

        # ToTensor() for uint16
        img_A = img_A/1.0 # torch.from_numpy: The only supported types are: float64, float32, float16, int64, int32, int16, int8, and uint8. uint16/1.0 -> float16
        img_A2 = torch.from_numpy((img_A.transpose((2, 0, 1)).copy())) # contiguous?
        # img_A2.float().div_(10000) # img_A.float().div(4096) # divide 10000 or 4096 or 65535?
        img_A2 = img_A2.clamp_(0, self.pixval_max)
        img_A2 = img_A2.float()/self.pixval_max # why can't I use img_A2.float().div_(10000)??????

        # Normalize() for float16
        mean = torch.as_tensor(np.full((self.channels,), 0.5), dtype=img_A2.dtype, device=img_A2.device)
        std = torch.as_tensor(np.full((self.channels,), 0.5), dtype=img_A2.dtype, device=img_A2.device)
        if mean.ndim == 1:
            mean = mean.view(-1, 1, 1)
        if std.ndim == 1:
            std = std.view(-1, 1, 1)
        img_A2.sub_(mean).div_(std)


        return {"A": img_A2, "name": filename_A}

    def __len__(self):
        return len(self.files_A)

class Satellitehidden_test_dataset_HazyGT(Dataset):
    def __init__(self, rootA, rootB, channels, pixval_max=10000):
        self.transform = transforms.Compose(
            [
                # transforms.ToTensor(),
                # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        self.files_A = sorted(glob.glob(rootA + "/*.*"))
        self.files_B = sorted(glob.glob(rootB + "/*.*"))
        self.channels = channels
        self.pixval_max = pixval_max/1.0

    def __getitem__(self, index):
        img_A, CRSA, TransformA, dTypeA = read(self.files_A[index % len(self.files_A)])
        img_B, CRSB, TransformB, dTypeB = read(self.files_B[index % len(self.files_B)])
        if not(CRSA==CRSB and TransformA==TransformB and dTypeA==dTypeB):
            print((CRSA, CRSB))
            print((TransformA,TransformB))
            print((dTypeA,dTypeB))
            sys.exit("Uh-oh...CRS are not the same for Hazy and GT.")
        # CRS, Transform, dType = CRSA, TransformA, dTypeA
        CRS, Transform, dType = CRSA, TransformA.to_gdal(), dTypeA

        if self.channels==3:
            img_A = img_A[:,:,0:3]
            img_B = img_B[:,:,0:3]

        path_A = self.files_A[index % len(self.files_A)]
        path_B = self.files_B[index % len(self.files_B)]
        filename_A = os.path.basename(path_A) 
        filename_B = os.path.basename(path_B) 

        # ToTensor() for uint16
        img_A = img_A/1.0 # torch.from_numpy: The only supported types are: float64, float32, float16, int64, int32, int16, int8, and uint8. uint16/1.0 -> float16
        img_A2 = torch.from_numpy((img_A.transpose((2, 0, 1)).copy())) # contiguous?
        img_A2 = img_A2.clamp_(0, self.pixval_max)
        img_A2 = img_A2.float()/self.pixval_max # why can't I use img_A2.float().div_(10000)??????

        # ToTensor() for uint16
        img_B = img_B/1.0 # torch.from_numpy: The only supported types are: float64, float32, float16, int64, int32, int16, int8, and uint8. uint16/1.0 -> float16
        img_B2 = torch.from_numpy((img_B.transpose((2, 0, 1)).copy())) # contiguous?
        img_B2 = img_B2.clamp_(0, self.pixval_max)
        img_B2 = img_B2.float()/self.pixval_max # why can't I use img_A2.float().div_(10000)??????

        # Normalize() for float16
        mean = torch.as_tensor(np.full((self.channels,), 0.5), dtype=img_A2.dtype, device=img_A2.device)
        std = torch.as_tensor(np.full((self.channels,), 0.5), dtype=img_A2.dtype, device=img_A2.device)
        if mean.ndim == 1:
            mean = mean.view(-1, 1, 1)
        if std.ndim == 1:
            std = std.view(-1, 1, 1)
        img_A2.sub_(mean).div_(std)
        img_B2.sub_(mean).div_(std)

        # https://rasterio.readthedocs.io/en/latest/quickstart.html
        # https://medium.com/@mommermiscience/dealing-with-geospatial-raster-data-in-python-with-rasterio-775e5ba0c9f5
        # return {"A": img_A2, "B": img_B2, "nameA": filename_A, "nameB": filename_B, "CRS": CRS, "dType": dType, "Transform": Transform * (0, 0)}
        return {"A": img_A2, "B": img_B2, "nameA": filename_A, "nameB": filename_B, "CRS": CRS, "dType": dType, "Transform": Transform}
        # return {"A": img_A2, "B": img_B2, "nameA": filename_A, "nameB": filename_B} #!

    def __len__(self):
        return len(self.files_A)

class classification(Dataset):
    def __init__(self, csv_path):
        self.transform = transforms.Compose(
            [
                transforms.Resize((299,299)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ]
        )
        
        self.GT = []
        self.Hazy = []
        self.label = []
        with open(csv_path, newline='') as csvfile:
            rows = csv.reader(csvfile)
            for row in rows:
                self.GT.append(row[0])
                self.Hazy.append(row[1])
                self.label.append(int(row[2]))

    def __getitem__(self, index):
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        img_gt = Image.open(self.GT[index % len(self.GT)])
        img_hazy = Image.open(self.Hazy[index % len(self.GT)])
        label = self.label[index % len(self.GT)]

        img_gt = self.transform(img_gt)
        img_hazy = self.transform(img_hazy)

        haze = img_hazy - img_gt

        return {"haze": haze, "label": label}

    def __len__(self):
        return len(self.GT)














