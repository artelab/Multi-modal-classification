#!/usr/bin/env bash

echo "Extracting encoded dataset..."
python3 scripts/extract_encoded_images.py extraction_parameters.csv
echo "Done!"
