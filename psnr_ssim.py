import numpy as np
import argparse
import os
from PIL import Image
from skimage.measure import compare_psnr, compare_ssim
import statistics
from natsort import natsorted, ns

# CURR_DIR = os.path.abspath(os.path.dirname(__file__)) #C:/.../Indoor/indoor
# python3 psnr_ssim.py --GT_dir --DH_dir

parser = argparse.ArgumentParser()
parser.add_argument('--GT_dir', required=False,
  default='./sample/GT_test',  help='')
parser.add_argument('--DH_dir', required=False,
  default='./indoor/selftrain_result500-3sub',  help='')
opt = parser.parse_args()

sys.stdout = open("psnr_ssim.txt", "w")

GT_dir = opt.GT_dir
DH_dir = opt.DH_dir
# GT_path = os.path.join(CURR_DIR,GT_dir)
# DH_path = os.path.join(CURR_DIR,DH_dir)
GT_path = GT_dir
DH_path = DH_dir

GTimg=[]

for root1, _, fnames in (os.walk(GT_path)):
  for i, fname1 in enumerate( natsorted(fnames, key=lambda y: y.lower()) ):
    img1 = Image.open(os.path.join(CURR_DIR, GT_dir, fname1)).convert('RGB') # img = Image.open(os.path.join(root, fname)).convert('RGB')
    GTimg.append(img1)

psnr_list=[]
ssim_list=[]
print('GT_dir=%s', opt.GT_dir)
print('DH_dir=%s', opt.DH_dir)
for i in range(0,5):
    GTimg_convert_ndarray = np.array(GTimg[i])
    DHimg = Image.open(os.path.join(DH_path,str(i)+'.png'))
    DHimg_convert_ndarray = np.array(DHimg)
    psnr = compare_psnr(GTimg_convert_ndarray,DHimg_convert_ndarray)
    ssim = compare_ssim(GTimg_convert_ndarray,DHimg_convert_ndarray,multichannel = True)
    psnr_list.append(psnr)
    ssim_list.append(ssim)
    print('%f' % (psnr))
    print('%f' % (ssim))
print('meanpsnr=%f' % (statistics.mean(psnr_list)) )
print('meanssim=%f' % (statistics.mean(ssim_list)) )
print('stdevpsnr=%f' % (statistics.stdev(psnr_list)) )
print('stdevssim=%f' % (statistics.stdev(ssim_list)) )
sys.stdout.close()
# for i in range(0,1):
#     GTimg = Image.open(os.path.join(GT_path,str(i)+'_indoor_GT.jpg'))
#     GTimg_convert_ndarray = np.array(GTimg)
#     DHimg = Image.open(os.path.join(DH_path,str(i)+'.png'))
#     DHimg_convert_ndarray = np.array(DHimg)
#     psnr = compare_psnr(GTimg_convert_ndarray,DHimg_convert_ndarray)
#     ssim = compare_ssim(GTimg_convert_ndarray,DHimg_convert_ndarray,multichannel = True)
#     print('PSNR %d: %f' % (i, psnr))
#     print('SSIM %d: %f\n' % (i, ssim))






