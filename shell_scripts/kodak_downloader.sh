#!/bin/bash

target_dir="/itet-stor/jparada/net_scratch/datasets"
mkdir -p $target_dir

curl -L -o $target_dir/kodak-dataset.zip https://www.kaggle.com/api/v1/datasets/download/sherylmehta/kodak-dataset
unzip $target_dir/kodak-dataset.zip -d $target_dir/kodak/
rm $target_dir/kodak-dataset.zip
