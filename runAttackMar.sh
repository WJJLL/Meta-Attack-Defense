#!/bin/bash

# Fixed Params
DATA_DIR='./data'
TARGET_DATASET='dukemtmc'
TARGET_NET_DIR='./pretrained_models/duke.pth'
SOURCE_DATASET='market1501'
SOURCE_NET_DIR='./pretrained_models/market.pth'
MAX_EPS=8
BATCH_SIZE=64
#Noise_TYPE 0:COLOR+DELTA; Noise_TYPE 1:COLOR; Noise_TYPE 2:DELTA
Noise_SAVE_DIR='./attackModel/marketIDE/TripletMeta'
VRITUAL_DATASETS='personx'
#VRITUAL_DATASETS='personxAll'
for ((Noise_TYPE=0; Noise_TYPE<=2; Noise_TYPE++)) do
  CUDA_VISIBLE_DEVICES=2,3 python -W ignore metaAttack.py \
      -t $TARGET_DATASET \
      -s $SOURCE_DATASET \
      -m $VRITUAL_DATASETS \
      --data $DATA_DIR \
      --resumeTgt $TARGET_NET_DIR \
      --resume $SOURCE_NET_DIR \
      --batch_size $BATCH_SIZE \
      --noise_type $Noise_TYPE \
      --combine-trainval \
      --noise_resume $Noise_SAVE_DIR'/'$Noise_TYPE >$SOURCE_DATASET'_'$Noise_TYPE'_meta.txt'

#      >$SOURCE_DATASET'_'$Noise_TYPE'_meta.txt'
done

