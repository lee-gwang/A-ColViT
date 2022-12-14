## Installation

```
conda create -n colorization python=3.9 -y
conda activate colorization
pip install -r requirements.txt
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```

## Preprocess
```
python preparation/make_mask.py --img_dir /home/data/imagenet/ctest10k/ --hint_dir ./data/ctest10k
```

## Training

```
bash scripts/train_pruned.sh
```


## Inference

```
bash scripts/infer_pruned.sh
```

## Demo
coming soon


## Acknowledgements
* [https://github.com/pmh9960/iColoriT](https://github.com/pmh9960/iColoriT)

* [https://github.com/richzhang/colorization-pytorch](https://github.com/richzhang/colorization-pytorch)
