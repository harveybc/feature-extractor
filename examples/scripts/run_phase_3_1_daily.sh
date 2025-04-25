#!/bin/bash

CONFIG_DIR="examples/config/phase_3_1_daily"

for file in "$CONFIG_DIR"/*.json; do
    echo "Running preprocessor with configuration: $(basename "$file")"
    sh ./predictor.sh --load_config "$file"
done

echo "All configurations processed."
