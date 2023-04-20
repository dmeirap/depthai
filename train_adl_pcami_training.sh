export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

OUTPUT_STRIDE=16
BATCH_SIZE=16
NUM_ITERATIONS=200000
INIT_CKPT="/home/adl/adl/training/deeplab/datasets/adl_pcami_training_etiquetas/exp/train_on_trainval_set/init_models/deeplabv3_adl_pcami_train_aug/model.ckpt-100000"
LOG_DIR_TRAIN="/home/adl/adl/training/deeplab/datasets/adl_pcami_training_etiquetas/exp/train_on_trainval_set/train"
DATASET_PATH="/home/adl/adl/training/deeplab/datasets/adl_pcami_training_etiquetas/tfrecord"
CROP_SIZE_TXT=240,135

python3 deeplab/train.py \
--logtostderr \
--train_split="train" \
--model_variant="mobilenet_v2" \
--output_stride=$OUTPUT_STRIDE \
--train_crop_size=$CROP_SIZE_TXT \
--train_batch_size=$BATCH_SIZE \
--training_number_of_steps=$NUM_ITERATIONS \
--fine_tune_batch_norm=true \
--tf_initial_checkpoint=$INIT_CKPT \
--train_logdir=$LOG_DIR_TRAIN \
--dataset="adl_pcami_240_50000" \
--dataset_dir=$DATASET_PATH

