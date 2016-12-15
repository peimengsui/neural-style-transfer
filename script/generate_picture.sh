

cd $SCRATCH/project/neural-style-transfer/
module purge
module load torch/gnu/20160623


JOBID=ttest
MODEL=la_muse

th fast_style_transfer.lua -cuda  \
 -use_model trained_model/model_$MODEL.t7 \
 -input_size 650 \
 -input_img input/picture/chicago.jpg \
 -output_img output/output_chicago_$JOBID.jpg




