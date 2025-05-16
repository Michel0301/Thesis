#!/bin/bash

# CESM: 1-shot tests

nvidia-smi --query-gpu=power.draw --format=csv -l 1 > /home/micber/spiking-meta-learning/testingRun/CESM/1shot5way/gpu_power_log_spiking_classfier_test_one_shot_1.txt &
pid=$!
python test_few_shot.py --shot 1 > /home/micber/spiking-meta-learning/testingRun/CESM/1shot5way/terminal_log_spiking_classfier_test_one_shot_1.txt 2>&1
kill $pid

nvidia-smi --query-gpu=power.draw --format=csv -l 1 > /home/micber/spiking-meta-learning/testingRun/CESM/1shot5way/gpu_power_log_spiking_classfier_test_one_shot_2.txt &
pid=$!
python test_few_shot.py --shot 1 > /home/micber/spiking-meta-learning/testingRun/CESM/1shot5way/terminal_log_spiking_classfier_test_one_shot_2.txt 2>&1
kill $pid

nvidia-smi --query-gpu=power.draw --format=csv -l 1 > /home/micber/spiking-meta-learning/testingRun/CESM/1shot5way/gpu_power_log_spiking_classfier_test_one_shot_3.txt &
pid=$!
python test_few_shot.py --shot 1 > /home/micber/spiking-meta-learning/testingRun/CESM/1shot5way/terminal_log_spiking_classfier_test_one_shot_3.txt 2>&1
kill $pid

# CESM: 5-shot tests 

nvidia-smi --query-gpu=power.draw --format=csv -l 1 > /home/micber/spiking-meta-learning/testingRun/CESM/5shot5way/gpu_power_log_spiking_classfier_test_five_shot_test_1.txt &
pid=$!
python test_few_shot.py --shot 5 > /home/micber/spiking-meta-learning/testingRun/CESM/5shot5way/terminal_log_spiking_classfier_test_five_shot_test_1.txt 2>&1
kill $pid

nvidia-smi --query-gpu=power.draw --format=csv -l 1 > /home/micber/spiking-meta-learning/testingRun/CESM/5shot5way/gpu_power_log_spiking_classfier_test_five_shot_test_2.txt &
pid=$!
python test_few_shot.py --shot 5 > /home/micber/spiking-meta-learning/testingRun/CESM/5shot5way/terminal_log_spiking_classfier_test_five_shot_test_2.txt 2>&1
kill $pid

nvidia-smi --query-gpu=power.draw --format=csv -l 1 > /home/micber/spiking-meta-learning/testingRun/CESM/5shot5way/gpu_power_log_spiking_classfier_test_five_shot_test_3.txt &
pid=$!
python test_few_shot.py --shot 5 > /home/micber/spiking-meta-learning/testingRun/CESM/5shot5way/terminal_log_spiking_classfier_test_five_shot_test_3.txt 2>&1
kill $pid

# MESM: train meta-learner
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64 \
python train_mesm.py --config configs/train_meta_mini.yaml > /home/micber/spiking-meta-learning/testingRun/MESM/train_log_meta.txt 2>&1

# MESM: 1-shot tests

nvidia-smi --query-gpu=power.draw --format=csv -l 1 > /home/micber/spiking-meta-learning/testingRun/MESM/1shot5way/gpu_power_log_spiking_meta_test_one_shot_1.txt &
pid=$!
python test_few_shot.py --shot 1 > /home/micber/spiking-meta-learning/testingRun/MESM/1shot5way/terminal_log_spiking_meta_test_one_shot_1.txt 2>&1
kill $pid

nvidia-smi --query-gpu=power.draw --format=csv -l 1 > /home/micber/spiking-meta-learning/testingRun/MESM/1shot5way/gpu_power_log_spiking_meta_test_one_shot_2.txt &
pid=$!
python test_few_shot.py --shot 1 > /home/micber/spiking-meta-learning/testingRun/MESM/1shot5way/terminal_log_spiking_meta_test_one_shot_2.txt 2>&1
kill $pid

nvidia-smi --query-gpu=power.draw --format=csv -l 1 > /home/micber/spiking-meta-learning/testingRun/MESM/1shot5way/gpu_power_log_spiking_meta_test_one_shot_3.txt &
pid=$!
python test_few_shot.py --shot 1 > /home/micber/spiking-meta-learning/testingRun/MESM/1shot5way/terminal_log_spiking_meta_test_one_shot_3.txt 2>&1
kill $pid

# MESM: 5-shot tests

nvidia-smi --query-gpu=power.draw --format=csv -l 1 > /home/micber/spiking-meta-learning/testingRun/MESM/5shot5way/gpu_power_log_spiking_meta_test_five_shot_test_1.txt &
pid=$!
python test_few_shot.py --shot 5 > /home/micber/spiking-meta-learning/testingRun/MESM/5shot5way/terminal_log_spiking_meta_test_five_shot_test_1.txt 2>&1
kill $pid

nvidia-smi --query-gpu=power.draw --format=csv -l 1 > /home/micber/spiking-meta-learning/testingRun/MESM/5shot5way/gpu_power_log_spiking_meta_test_five_shot_test_2.txt &
pid=$!
python test_few_shot.py --shot 5 > /home/micber/spiking-meta-learning/testingRun/MESM/5shot5way/terminal_log_spiking_meta_test_five_shot_test_2.txt 2>&1
kill $pid

nvidia-smi --query-gpu=power.draw --format=csv -l 1 > /home/micber/spiking-meta-learning/testingRun/MESM/5shot5way/gpu_power_log_spiking_meta_test_five_shot_test_3.txt &
pid=$!
python test_few_shot.py --shot 5 > /home/micber/spiking-meta-learning/testingRun/MESM/5shot5way/terminal_log_spiking_meta_test_five_shot_test_3.txt 2>&1
kill $pid
