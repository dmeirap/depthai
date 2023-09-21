export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

LOG_DIR_TRAIN="/home/adl/adl/training/deeplab/datasets/mcr_seg/exp/train_on_trainval_set/train"
DATASET_PATH="/home/adl/adl/training/deeplab/datasets/mcr_seg/tfrecord"
LOG_DIR_VIS="/home/adl/adl/training/deeplab/datasets/mcr_seg/exp/train_on_trainval_set/vis"

python3 deeplab/vis.py \
--logtostderr \
--vis_split="val" \
--model_variant="mobilenet_v2" \
--vis_crop_size="240,240" \
--checkpoint_dir=$LOG_DIR_TRAIN \
--vis_logdir=$LOG_DIR_VIS \
--dataset="mcr_seg" \
--dataset_dir=$DATASET_PATH \
--max_number_of_iterations=1
