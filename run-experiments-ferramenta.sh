#!/usr/bin/env bash

echo "Downloading the dataset..."
wget http://artelab.dista.uninsubria.it/downloads/datasets/commercial_offers/ferramenta/images-val.tar.gz
wget http://artelab.dista.uninsubria.it/downloads/datasets/commercial_offers/ferramenta/images-train.tar.gz
wget http://artelab.dista.uninsubria.it/downloads/datasets/commercial_offers/ferramenta/text-train.tar.gz
wget http://artelab.dista.uninsubria.it/downloads/datasets/commercial_offers/ferramenta/text-val.tar.gz

echo "Uncompressing the dataset..."
tar -xvfz images-val.tar.gz
tar -xvfz images-train.tar.gz
tar -xvfz text-train.tar.gz
tar -xvfz text-val.tar.gz

echo "Starting training process..."
python3 train_model.py training_flags.csv
echo "Done!"
