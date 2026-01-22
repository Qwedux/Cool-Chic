#!/bin/bash
# filepath: run_encode_parallel.sh

start_time=$(date +%s)

# Run with maximum 4 parallel jobs
seq 0 13 | parallel -j 4 \
  python3 lossless_encode.py \
    --image-index={} \
    --color-space=YCoCg \
    --encoder-gain=64 \
    --use-image-arm \
    --experiment_name=2026_01_18_assorted_dataset_test

end_time=$(date +%s)
elapsed=$((end_time - start_time))

echo "================================"
echo "All processes completed in ${elapsed} seconds"
echo "================================"