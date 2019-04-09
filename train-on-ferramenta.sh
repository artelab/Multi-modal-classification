#!/usr/bin/env bash


declare -a arr=("http://artelab.dista.uninsubria.it/downloads/datasets/commercial_offers/ferramenta/images-train.tar.gz"
                "http://artelab.dista.uninsubria.it/downloads/datasets/commercial_offers/ferramenta/images-val.tar.gz"
                "http://artelab.dista.uninsubria.it/downloads/datasets/commercial_offers/ferramenta/text-train.tar.gz"
                "http://artelab.dista.uninsubria.it/downloads/datasets/commercial_offers/ferramenta/text-val.tar.gz")

for link in "${arr[@]}"
do
   file="$(basename ${link})"
   if [[ -f "$file" ]]
   then
        echo "$file found. Skipping"
   else
        echo "Downloading the dataset..."
        curl -O ${link}
        tar -xvfz ${file}
    fi
done

echo "Starting training process..."
python3 train_model.py training_flags.csv
echo "Done!"
