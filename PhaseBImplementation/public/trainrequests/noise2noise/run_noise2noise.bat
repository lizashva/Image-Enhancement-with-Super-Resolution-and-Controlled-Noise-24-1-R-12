@echo off
 python ./train.py ^
    --train-dir "dataset/data_set2/train" ^
    --valid-dir "dataset/data_set2/valid" ^
    --ckpt-save-path "./ckpts" ^
    --ckpt-overwrite ^
    --train-size 1000 ^
    --valid-size 300 ^
    --nb-epochs 50 ^
    --loss l2 ^
    --noise-type multiplicative bernoulli ^
    --noise-param 25 ^
    --crop-size 256
