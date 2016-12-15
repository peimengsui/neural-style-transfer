#!/bin/bash
#PBS -l nodes=1:ppn=1:gpus=1
#PBS -l walltime=12:00:00
#PBS -l mem=4GB
#PBS -m abe
#PBS -M mp4504@nyu.edu
#PBS -j oe
#PBS -N ttest

module purge
module load torch/gnu/20160623
RUNDIR=$SCRATCH/project/neural-style-transfer/
cd $RUNDIR/

H5PATH=/scratch/mp4504/coco/hdf5_file/coco80k.h5
JOBID=ttest
CONTENT_WEIGHT=2.0
TV_WEIGHT=1e-5
TRAIN_SIZE=100
STYLE=input/art/mosaic.jpg


echo $JOBID  style: $STYLE
echo vgg16 instead of vgg19
echo $TRAIN_SIZE training data, contentLoss and styleLoss modified
echo contentLoss weight set to $CONTENT_WEIGHT
echo TVloss weight: $TV_WEIGHT

th train_model.lua -style_image $STYLE  \
 -h5_file $H5PATH \
 -max_train $TRAIN_SIZE \
 -jobid $JOBID \
 -cuda -content_weight $CONTENT_WEIGHT -TV_weight $TV_WEIGHT

th fast_style_transfer.lua -cuda  \
 -use_model trained_model/model_$JOBID.t7 \
 -input_size 650 \
 -output_img output/output_tubigen_$JOBID.jpg

th fast_style_transfer.lua -cuda  \
 -use_model trained_model/model_$JOBID.t7 \
 -input_size 650 \
 -input_img input/picture/chicago.jpg \
 -output_img output/output_chicago_$JOBID.jpg



