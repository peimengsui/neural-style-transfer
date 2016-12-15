# Neural Style Transfer

Make sure to download VGG-19 (for slow verion) and VGG-16 (for fast version) and put them in the /model directory
before generating art picture (slow version) or training an image transformation model (fast version).

Enable cuda support for faster computation.

## Requirement

- torch
- loadcaffe

For Cuda backend:

- cunn
- cudnn
- cutorch

Make sure to download VGG-19 before running the script.

This can be done by running:
For slow version:
```{bash}
sh models/download_models.sh
```
For fast version:
```{bash}
sh models/download_vgg16.sh
```

Basic usage:

Generate an art image using slow version:

    th slow-style-transfer.lua -cuda 
 
  Output image will be put in the /output directory.
  
Generate an art image using fast version:

    th fast-style-transfer.lua -cuda \
       -use_model PATH/TO/TRAINED_MODEL \ 
       -input_img PATH/TO/INPUT_IMAGE.jpg \
       -output_img PATH/TO/OUTPUT_IMAGE.jpg \
  
Train an image transformation model (for fast version only):

    th train_model.lua -cuda \ 
       -h5_file PATH/TO/TRAINING_DATASET \
       -max_train 20000 \
       -content_weight 2.0 -style_weight 5.0 \ 
       -TV_weight 1e-5 \
       -jobid default 
