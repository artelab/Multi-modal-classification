#!/usr/bin/env bash

w=10
encoding_height=1

echo "output_image_width $w"
echo "encoding_height $encoding_height"
python3 train.py --train_path "${HOME}/datasets/food-101/train.csv"  \
                 --val_path "${HOME}/datasets/food-101/test.csv"  \
                 --output_image_width $w \
                 --encoding_height ${encoding_height} \
                 --save_model_dir_name "runs/food101-$w-$encoding_height" \
                 --gpu_id "0" \
                 --patience 10 \
                 --batch_size 64 \
                 --num_epochs 200 \
                 --evaluate_every 500 \
                 --num_checkpoints 1

