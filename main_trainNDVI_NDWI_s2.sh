#!/bin/bash/

gpuid=$1
src_dir=$2
epoch=$3
n_epochs=$4
g_lr=$5
batch_size=$6
model=$7
patchsz=$8
pretrained_model_pathG=$9

echo "**************************************************"
echo "*  1st Stage: training *"
echo "**************************************************"
# how to use this script?
# sh main_trainNDVI_NDWI_s2.sh 0 L1C_England_augmented 0 80 0.0001 8 NTIRE2018_Multiscale_NDVI_NDWI 512
# sh main_trainNDVI_NDWI_s2.sh 0 20210303_augmented_11sites_moredetailed 0 80 0.0001 8 NTIRE2018_Multiscale_NDVI_NDWI 512
# sh main_trainNDVI_NDWI_s2.sh 2 20210303_crop30_augmented 0 80 0.0001 16 NTIRE2018_Multiscale_NDVI_NDWI 512

CUDA_VISIBLE_DEVICES=$gpuid python3 train_NDVI_NDWI_s2.py --gpuid $gpuid --dataset_name $src_dir --epoch $epoch --n_epochs $n_epochs --g_lr $g_lr --batch_size $batch_size --model $model --img_height $patchsz 

