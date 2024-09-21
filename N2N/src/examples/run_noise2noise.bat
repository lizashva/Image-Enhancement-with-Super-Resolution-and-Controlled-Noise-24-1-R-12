@echo off
 python ./train.py ^
    --train-dir "data/train" ^
    --valid-dir "data/valid" ^
    --ckpt-save-path "./ckpts" ^
    --ckpt-overwrite ^
    --train-size 500 ^
    --valid-size 100 ^
    --nb-epochs 30 ^
    --loss l1 ^
    --noise-type poisson ^
    --noise-param 25 ^
    --crop-size 256
