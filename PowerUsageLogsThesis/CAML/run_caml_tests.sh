#!/bin/bash

# CAML: 1-shot 5-way tests

nvidia-smi --query-gpu=power.draw --format=csv -l 1 > /home/micber/CAMLprogramStructure/logs/gpu_power_log_caml_1shot_1.txt &
pid=$!
python src/evaluation/test.py \
  --model CAML \
  --eval_dataset mini_ImageNet \
  --fe_type timm:vit_base_patch16_clip_224.openai:768 \
  > /home/micber/CAMLprogramStructure/logs/terminal_log_caml_1shot_1.txt 2>&1
kill $pid

nvidia-smi --query-gpu=power.draw --format=csv -l 1 > /home/micber/CAMLprogramStructure/logs/gpu_power_log_caml_1shot_2.txt &
pid=$!
python src/evaluation/test.py \
  --model CAML \
  --eval_dataset mini_ImageNet \
  --fe_type timm:vit_base_patch16_clip_224.openai:768 \
  > /home/micber/CAMLprogramStructure/logs/terminal_log_caml_1shot_2.txt 2>&1
kill $pid

nvidia-smi --query-gpu=power.draw --format=csv -l 1 > /home/micber/CAMLprogramStructure/logs/gpu_power_log_caml_1shot_3.txt &
pid=$!
python src/evaluation/test.py \
  --model CAML \
  --eval_dataset mini_ImageNet \
  --fe_type timm:vit_base_patch16_clip_224.openai:768 \
  > /home/micber/CAMLprogramStructure/logs/terminal_log_caml_1shot_3.txt 2>&1
kill $pid

# CAML: 5-shot 5-way tests 

nvidia-smi --query-gpu=power.draw --format=csv -l 1 > /home/micber/CAMLprogramStructure/logs/gpu_power_log_caml_5shot_1.txt &
pid=$!
python src/evaluation/test.py \
  --model CAML \
  --eval_dataset mini_ImageNet \
  --fe_type timm:vit_base_patch16_clip_224.openai:768 \
  > /home/micber/CAMLprogramStructure/logs/terminal_log_caml_5shot_1.txt 2>&1
kill $pid

nvidia-smi --query-gpu=power.draw --format=csv -l 1 > /home/micber/CAMLprogramStructure/logs/gpu_power_log_caml_5shot_2.txt &
pid=$!
python src/evaluation/test.py \
  --model CAML \
  --eval_dataset mini_ImageNet \
  --fe_type timm:vit_base_patch16_clip_224.openai:768 \
  > /home/micber/CAMLprogramStructure/logs/terminal_log_caml_5shot_2.txt 2>&1
kill $pid

nvidia-smi --query-gpu=power.draw --format=csv -l 1 > /home/micber/CAMLprogramStructure/logs/gpu_power_log_caml_5shot_3.txt &
pid=$!
python src/evaluation/test.py \
  --model CAML \
  --eval_dataset mini_ImageNet \
  --fe_type timm:vit_base_patch16_clip_224.openai:768 \
  > /home/micber/CAMLprogramStructure/logs/terminal_log_caml_5shot_3.txt 2>&1
kill $pid
