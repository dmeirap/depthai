export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

LOG_DIR_TRAIN="/home/adl/adl/training/deeplab/datasets/adl_dhl_mcr_seg/exp/train_on_trainval_set/train"
DATASET_PATH="/home/adl/adl/training/deeplab/datasets/adl_dhl_mcr_seg/tfrecord"
LOG_DIR_EVAL="/home/adl/adl/training/deeplab/datasets/adl_dhl_mcr_seg/exp/train_on_trainval_set/eval"

python3 deeplab/eval.py \
--logtostderr \
--eval_split="val" \
--model_variant="mobilenet_v2" \
--eval_crop_size="240,240" \
--checkpoint_dir=$LOG_DIR_TRAIN \
--eval_logdir=$LOG_DIR_EVAL \
--dataset="adl_dhl_mcr_seg" \
--dataset_dir=$DATASET_PATH \
--max_number_of_evaluations=1
