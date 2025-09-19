#!/bin/bash

# The assumption is that the script is run from the `Cool-Chic/coolchic` directory
# I.e. directories like `Cool-Chic/cfg` are accessible via `../cfg`
# the idea is that I want to run `python3 lossless_encode.py`. Upon running the script loads file index from `../cfg/img_index.txt`
# I want to go over all the images in the kodak dataset (24 images), so before each call of python3 lossless_encode.py I change the index in `../cfg/img_index.txt`
# Also for parallelization, I want to run odd or even image indexes only, so the script takes one argument: 0 or 1

if [ "$#" -ne 1 ]; then
    echo "Illegal number of parameters. Please provide 0 or 1 as the only parameter."
    exit 1
fi

if [ "$1" -ne 0 ] && [ "$1" -ne 1 ]; then
    echo "The only allowed parameters are 0 or 1."
    exit 1
fi

for i in {1..23}
do
    if [ $((i % 2)) -eq "$1" ]; then
        echo "Processing image index: $i"
        echo $i > ../cfg/img_index.txt
        python3 lossless_encode.py
    fi
done
