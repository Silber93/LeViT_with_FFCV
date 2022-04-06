# LeViT with FFCV
In this project, we implement and experiment the combination LeViT - Vision Transformer for fast inference with 
FFCV - Fast Forward Computer Vision  drop-in data loading system, which results a faster LeViT, in both training and inference, 
and reaches better accuracies with smaller scale architectures. <br>
## Installation
First, clone the repository to your machine.
```
git clone https://github.com/Silber93/LeViT_with_FFCV.git
```
Then, create a fresh new env that will be compatible with FFCV
```
conda create -y -n ffcv python=3.9 cupy pkg-config compilers libjpeg-turbo opencv pytorch torchvision cudatoolkit=11.3 numba -c pytorch -c conda-forge
conda activate ffcv
pip install ffcv
pip install timm
pip install matplotlib
```
## Usage
### Speed Test
speed_test.py is a standalone script that compares the speed (images/s) of 'clean' model architectures - original LeViT vs. LeViT with FFCV data loading. <br>
run:
```
srun --gres=gpu:1 python3 speed_test.py --ffcv-data-source-dir <source dir for FFCV file creation>
```
**NOTE** FFCV .dat file is required to perform speed tests or training.

### FFCV Data Creation
In order to use FFCV loading functinality, we need to write an FFCV .dat file that will be used as the data source.
```
srun --gres=gpu:1 python3 ffcv_writer.py --data-dir <data source dir> --split <train/val> --write-dir-name <.dat destination dir>
```

### Training
```
srun --gres=gpu:1 python3 main.py --output_dir saved_models/s_396 --data-path ~/data/imagenette2 --model LeViT_128S --eval-freq 2
```
* Select the model to train with the argument '--model' 
* Choose whether to use FFCV data loading with the argument '--ffcv-load' <[Y/N]> ('Y' by default)
