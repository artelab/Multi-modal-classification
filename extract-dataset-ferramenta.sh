#!/usr/bin/env bash

echo "Extracting encoded dataset..."
python3 extract_encoded_images.py extraction_flags.csv
echo "Done!"
