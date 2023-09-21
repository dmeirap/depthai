export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

OUTPUT_STRIDE=16
BATCH_SIZE=16
CKPT_PATH="/home/adl/adl/training/deeplab/datasets/mcr_seg/exp/train_on_trainval_set/train/model.ckpt-100000"
EXPORT_PATH="/home/adl/adl/training/deeplab/datasets/mcr_seg/exp/train_on_trainval_set/export/frozen_mcr_seg.pb"

python3 deeplab/export_model.py \
--logtostderr \
--checkpoint_path=$CKPT_PATH \
--export_path=$EXPORT_PATH \
--model_variant="mobilenet_v2" \
--inference_scales=1.0
