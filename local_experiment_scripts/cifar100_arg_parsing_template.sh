#!/bin/sh

cd ..
export DATASET_DIR="data/"
# Activate the relevant virtual environment:

python train_evaluate_emnist_classification_system.py --batch_size 100 --continue_from_epoch -1 --seed 0 \
                                                      --image_num_channels 3 --image_height 32 --image_width 32 \
                                                      --dim_reduction_type "strided" --num_layers 4 --num_filters 64 \
                                                      --num_epochs 100 --experiment_name 'cifar100_test_exp' \
                                                      --use_gpu "True" --weight_decay_coefficient 0. \
                                                      --dataset_name "cifar100"
