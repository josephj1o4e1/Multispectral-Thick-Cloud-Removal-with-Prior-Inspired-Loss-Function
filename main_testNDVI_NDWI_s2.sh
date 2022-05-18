# #!/bin/bash/

# How to use this script? For example, 
# sh main_test.sh Quicktest 1024 4 5 "Please_type_model_absolute_path_here"

# sh main_testNDVI_NDWI_s2.sh 20210303/val 1024 4 5 NTIRE2018_Multiscale_NDVI_NDWI /home/sychuang0909/sychuang0909_4journal/Code/saved_models_mgpu/Satellite4in4out_NTIRE2018_Multiscale_NDVI_NDWI/20210303_augmented/Mar12_12-12-23/generator_28.pth

datasetname=$1 # 20210303/val
patchsz=$2 # 512,1024,2048
model=$3
model_dir=$4 # model_absolute_path! 


echo "**************************************************"
echo "*  1st Stage: Crop test data to 1024*1024 patches  *"
echo "**************************************************"
python3.6 noresize_crop_shift_s2.py \
--datasetname $datasetname \
--L $patchsz 

echo "**************************************************"
echo "*           2nd Stage: Dehazing.....             *"
echo "**************************************************"
CUDA_VISIBLE_DEVICES=0 python3 test_NDVI_NDWI_s2.py \
--model_dir $model_dir \
--model $model \
--datasetname $datasetname \
--L $patchsz

echo "**************************************************"
echo "*      3rd Stage: Merging dehazed patches        *"
echo "**************************************************"
python3 noresize_merge_s2.py \
--model_dir $model_dir \
--model $model \
--datasetname $datasetname \
--L $patchsz
