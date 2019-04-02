#!/usr/bin/env bash

patience=20
w=10 # embedding width
encoding_height=1 # embedding height
ste_image_w=227
ste_separator_size=4
ste_superpixel_size=3

echo "output_image_width $w"
echo "encoding_height $encoding_height"
python3 extract_new_encoding.py \
                 --train_path "/home/superior/datasets/amazon_multimodal/train.csv"  \
                 --val_path "/home/superior/datasets/amazon_multimodal/val.csv"  \
                 --output_dir "/home/superior/datasets/amazon_multimodal/new-encoding/${w}x${w}-${encoding_height}-sep${ste_separator_size}-spixel${ste_superpixel_size}" \
                 --output_image_width $w \
                 --encoding_height $encoding_height \
                 --save_model_dir_name "runs/amazon-$w-$encoding_height" \
                 --gpu_id "" \
         --batch_size 64 \
         --ste_image_w ${ste_image_w} \
         --ste_separator_size ${ste_separator_size} \
         --ste_superpixel_size ${ste_superpixel_size}
