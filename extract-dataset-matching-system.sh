#!/usr/bin/env bash

# train_path = "/home/super/PycharmProjects/amazon_multimodal/train.csv"
# val_path = "/home/super/PycharmProjects/amazon_multimodal/val.csv"
# train_path = "/home/super/datasets/food101/train.csv"
# val_path = "/home/super/datasets/food101/test.csv"
# train_path = "/home/super/datasets/ferramenta52-multimodal/train.csv"
# val_path = "/home/super/datasets/ferramenta52-multimodal/val.csv"
# train_path = "/home/super/datasets/7pixel/matching-system/dataset/train/train.csv"
# val_path = "/home/super/datasets/7pixel/matching-system/dataset/train/train.csv"

patience=20
w=10 # embedding width
encoding_height=1 # embedding height
ste_image_w=256
ste_separator_size=4
ste_superpixel_size=3

#for w in 50 100 200; do
#    for encoding_height in 5 10 20; do
        echo "output_image_width $w"
        echo "encoding_height $encoding_height"
        python3 extract_new_encoding.py \
                         --train_path "/home/superior/datasets/7pixel/matching-system/dataset/train/train.csv"  \
                         --val_path "/home/superior/datasets/7pixel/matching-system/dataset/test/test.csv"  \
                         --output_dir "/home/superior/datasets/7pixel/matching-system/new-encoding/${w}x${w}-${encoding_height}-sep${ste_separator_size}-spixel${ste_superpixel_size}" \
                         --output_image_width $w \
                         --encoding_height $encoding_height \
                         --save_model_dir_name "runs/matching-system-$w-$encoding_height" \
                         --gpu_id "" \
                         --batch_size 64 \
                         --ste_image_w ${ste_image_w} \
                         --ste_separator_size ${ste_separator_size} \
                         --ste_superpixel_size ${ste_superpixel_size}

#    done
#done