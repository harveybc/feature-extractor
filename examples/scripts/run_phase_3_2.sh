#!/bin/bash

CONFIG_DIR="examples/config/phase_3_2"

for file in "$CONFIG_DIR"/*.json; do
    echo "Running feature extractor with configuration: $(basename "$file")"
    sh ./feature-extractor.sh --load_config "$file"
done

CONFIG_DIR="examples/config/phase_3_2_daily"

for file in "$CONFIG_DIR"/*.json; do
    echo "Running feature extractor with configuration: $(basename "$file")"
    sh ./feature-extractor.sh --load_config "$file"
done


echo "All configurations processed."
