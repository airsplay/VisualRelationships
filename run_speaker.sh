#!/bin/sh

# dataset from 'nlvr2', 'adobe', 'spotdiff'
dataset=adobe

# Main metric to use
# One of 'BLEU', 'METEOR', 'ROUGE_L', 'CIDEr', 'Bleu_1'~'Bleu_4', 'F1' (F1 is the f1 score of BLEU and ROUGE_L)
metric=CIDEr

# model from 'init', 'newheads', 'newcross', 'dynamic', which are the four model in paper 
# related to the four subsections in the paper
model=dynamic

# Name of the model, used in snapshot
name=${model}_2pixel

task=speaker
if [ -z "$1" ]; then
    gpu=0
else
    gpu=$1
fi

log_dir=$dataset/$task/$name
mkdir -p snap/$log_dir
mkdir -p log/$dataset/$task
cp $0 snap/$log_dir/run.bash
cp -r src snap/$log_dir/src

CUDA_VISIBLE_DEVICES=$gpu stdbuf -i0 -o0 -e0 python src/main.py --output snap/$log_dir \
    --maxInput 40 --metric $metric --model $model --imgType pixel --worker 4 --train speaker --dataset $dataset \
    --batchSize 95 --hidDim 512 --dropout 0.5 \
    --seed 9595 \
    --optim adam --lr 1e-4 --epochs 500 \
    | tee log/$log_dir.log
