#!/usr/bin/env bash


declare -a arr=("http://artelab.dista.uninsubria.it/downloads/datasets/commercial_offers/ferramenta/images-train.tar.gz"
                "http://artelab.dista.uninsubria.it/downloads/datasets/commercial_offers/ferramenta/images-val.tar.gz"
                "http://artelab.dista.uninsubria.it/downloads/datasets/commercial_offers/ferramenta/text-image-train.csv"
                "http://artelab.dista.uninsubria.it/downloads/datasets/commercial_offers/ferramenta/text-image-val.csv")

for link in "${arr[@]}"
do
   file="$(basename ${link})"
   if [[ -f "$file" ]]
   then
        echo "$file found. Skipping"
   else
        echo "Downloading $file"
        curl -O ${link}
   fi
done

echo "Unpacking images files..."
tar xfz images-train.tar.gz
tar xfz images-val.tar.gz

echo "Renaming files..."
mv images-train train
mv images-val val
mv text-image-train.csv train.csv
mv text-image-val.csv val.csv

echo "Starting training process..."
python3 scripts/train_model.py training_parameters.csv
echo "Done!"
