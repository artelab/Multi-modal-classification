#!/usr/bin/env bash

# train_path = "/home/super/PycharmProjects/amazon_multimodal/train.csv"
# val_path = "/home/super/PycharmProjects/amazon_multimodal/val.csv"
# train_path = "/home/super/datasets/food101/train.csv"
# val_path = "/home/super/datasets/food101/test.csv"
# train_path = "/home/super/datasets/ferramenta52-multimodal/train.csv"
# val_path = "/home/super/datasets/ferramenta52-multimodal/val.csv"
# train_path = "/home/super/datasets/7pixel/matching-system/dataset/train/train.csv"
# val_path = "/home/super/datasets/7pixel/matching-system/dataset/train/train.csv"

w=10
encoding_height=1

#for w in 50 100 200; do
#    for encoding_height in 5 10 20; do
        echo "output_image_width $w"
        echo "encoding_height $encoding_height"
        python3 train.py --train_path "/home/superior/datasets/7pixel/matching-system/dataset/train/train.csv"  \
                         --val_path "/home/superior/datasets/7pixel/matching-system/dataset/test/test.csv"  \
                         --output_image_width $w \
                         --encoding_height $encoding_height \
                         --save_model_dir_name "runs/matching-system-$w-$encoding_height" \
                         --gpu_id "0" \
                         --patience 10 \
                         --batch_size 64 \
                         --num_epochs 200 \
                         --evaluate_every 500 \
                         --num_checkpoints 1 \

#    done
#done