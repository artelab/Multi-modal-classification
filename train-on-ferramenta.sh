#!/usr/bin/env bash


declare -a arr=("http://artelab.dista.uninsubria.it/downloads/datasets/commercial_offers/ferramenta/images-train.tar.gz"
                "http://artelab.dista.uninsubria.it/downloads/datasets/commercial_offers/ferramenta/images-val.tar.gz"
                "http://artelab.dista.uninsubria.it/downloads/datasets/commercial_offers/ferramenta/text-images-train.csv"
                "http://artelab.dista.uninsubria.it/downloads/datasets/commercial_offers/ferramenta/text-images-val.csv")

for link in "${arr[@]}"
do
   file="$(basename ${link})"
   if [[ -f "$file" ]]
   then
        echo "$file found. Skipping"
   else
        echo "Downloading the dataset..."
        curl -O ${link}
   fi
done

if [[ -f images-train ]]
then
    tar xfz images-train.tar.gz
    mv images-train train
fi

if [[ -f images-val ]]
then
    tar xfz images-val.tar.gz
    mv images-val val
fi

mv text-images-train.csv train.csv
mv text-images-val.csv val.csv

echo "Starting training process..."
python3 scripts/train_model.py training_parameters.csv
echo "Done!"
