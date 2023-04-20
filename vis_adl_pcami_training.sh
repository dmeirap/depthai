export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

LOG_DIR_TRAIN="/home/adl/adl/training/deeplab/datasets/adl_pcami_training_etiquetas/exp/train_on_trainval_set/train"
DATASET_PATH="/home/adl/adl/training/deeplab/datasets/adl_pcami_training_etiquetas/tfrecord"
LOG_DIR_VIS="/home/adl/adl/training/deeplab/datasets/adl_pcami_training_etiquetas/exp/train_on_trainval_set/vis"

python3 deeplab/vis.py \
--logtostderr \
--vis_split="val" \
--model_variant="mobilenet_v2" \
--vis_crop_size="240,240" \
--checkpoint_dir=$LOG_DIR_TRAIN \
--vis_logdir=$LOG_DIR_VIS \
--dataset="adl_pcami_240_50000" \
--dataset_dir=$DATASET_PATH \
--max_number_of_iterations=1
