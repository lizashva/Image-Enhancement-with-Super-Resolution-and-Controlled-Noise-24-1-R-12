# Noise2Noise: Learning Image Restoration without Clean Data

This is an unofficial PyTorch implementation of [Noise2Noise](https://arxiv.org/abs/1803.04189) (Lehtinen et al. 2018).

## Dependencies

* [PyTorch](https://pytorch.org/) (0.4.1)
* [Torchvision](https://pytorch.org/docs/stable/torchvision/index.html) (0.2.0)
* [NumPy](http://www.numpy.org/) (1.14.2)
* [Matplotlib](https://matplotlib.org/) (2.2.3)
* [Pillow](https://pillow.readthedocs.io/en/latest/index.html) (5.2.0)
* [OpenEXR](http://www.openexr.com/) (1.3.0)

To install the latest version of all packages, run
```
pip3 install --user -r requirements.txt
```

This code was tested on Python 3.9 

## Dataset

You can download the full dataset by running 'LoadDataSet.bat' or use another dataset. Add your dataset to 'data' folder when the train images insert to 'train' folder and validation images to 'valid' folder.
```
mkdir data
cd data
mkdir train valid test
curl -O http://images.cocodataset.org/zips/test2017.zip
curl -O http://images.cocodataset.org/zips/val2017.zip
tar -xf test2017.zip -C train --strip-components=1
tar -xf val2017.zip -C valid --strip-components=1
```


## Training

See `python3 train.py --h` for list of optional arguments, or `examples/train.bat` for an example.

By default, the model train with noisy targets. To train with clean targets, use `--clean-targets`. To train and validate on smaller datasets, use the `--train-size` and `--valid-size` options. To plot stats as the model trains, use `--plot-stats`; these are saved alongside checkpoints. By default CUDA is not enabled: use the `--cuda` option if you have a GPU that supports it.

### Gaussian noise
The noise parameter is the maximum standard deviation σ.
```
python3 train.py ^
  --train-dir ../data/train --train-size 1000 ^
  --valid-dir ../data/valid --valid-size 200 ^
  --ckpt-save-path ../ckpts ^
  --nb-epochs 10 ^
  --batch-size 4 ^
  --loss l2 ^
  --noise-type gaussian ^
  --noise-param 50 ^
  --crop-size 128 ^
  --plot-stats ^
  --cuda
```

### Poisson noise
The noise parameter is the Poisson parameter λ.
```
python3 train.py
  --loss l2 ^
  --noise-type poisson ^
  --noise-param 25 ^
  --cuda
```

### Text overlay
The noise parameter is the approximate probability *p* that a pixel is covered by text.
```
python3 train.py ^
  --loss l1 ^
  --noise-type text ^
  --noise-param 1.2 ^
  --cuda
```

## Testing

Model checkpoints are automatically saved after every epoch. To test the denoiser, provide `test.py` with a PyTorch model (`.pt` file) via the argument `--load-ckpt` and a test image directory via `--data`. The `--show-output` option specifies the number of noisy/denoised/clean montages to display on screen. To disable this, simply remove `--show-output`.

```
python ./test.py ^
--data ./data/test ^
--load-ckpt ./ckpts/gaussian/n2n-gaussian.pt ^
--noise-type gaussian ^
--noise-param 50 ^
--seed 1 ^
--crop-size 256 ^
--show-output 2 
```

See `python3 test.py --h` for list of optional arguments, or `examples/test.bat` for an example.

## Results

You can see our results in the folder 'Results'.

## References
* Jaakko Lehtinen, Jacob Munkberg, Jon Hasselgren, Samuli Laine, Tero Karras, Miika Aittala,and Timo Aila. [*Noise2Noise: Learning Image Restoration without Clean Data*](https://research.nvidia.com/publication/2018-07_Noise2Noise%3A-Learning-Image). Proceedings of the 35th International Conference on Machine Learning, 2018.
