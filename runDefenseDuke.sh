#!/bin/bash

# Fixed Params
DATA_DIR='./data'
SOURCE_DATASET='dukemtmc'
SOURCE_NET_DIR='./pretrained_models/duke.pth'
MAX_EPS=8
BATCH_SIZE=64
#Noise_TYPE 0:COLOR+DELTA; Noise_TYPE 1:COLOR; Noise_TYPE 2:DELTA
VRITUAL_DATASETS='personx'
#VRITUAL_DATASETS='personxAll'
EPOCHS=85
Noise_MODEL_Path='./attackModel/dukeIDE/'
REID_MODEL_SAVE_DIR='./defenseModel/duke/'
for ((Noise_TYPE=0; Noise_TYPE<=2; Noise_TYPE++)) do
  CUDA_VISIBLE_DEVICES=0,1 python -W ignore defenseMaml.py \
      -s $SOURCE_DATASET \
      -m $VRITUAL_DATASETS \
      --data $DATA_DIR \
      --resume $SOURCE_NET_DIR \
      --batch_size $BATCH_SIZE \
      --noise_type $Noise_TYPE \
      --combine-trainval \
      --epochs $EPOCHS \
      --max-eps $MAX_EPS \
      --noise_path1 $Noise_MODEL_Path$Noise_TYPE'/Tripletpp/best_perturbation4.pth' \
      --logs_dir $REID_MODEL_SAVE_DIR'/'$Noise_TYPE > $SOURCE_DATASET'_'$Noise_TYPE'_defense.txt'
done

