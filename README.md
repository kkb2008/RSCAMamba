# RSCAMamba


## Install

Open the folder **airs** using **Linux Terminal** and create python environment:
```
conda create -n airs python=3.8
conda activate airs
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r GeoSeg/requirements.txt
```

Install Mamba
```
pip install causal-conv1d>=1.4.0
pip install mamba-ssm
```

## Training

"-c" means the path of the config, use different **config** to train different models.

```
python GeoSeg/train_supervision.py -c GeoSeg/config/uavid/unetformer.py
```

## Testing

"-c" denotes the path of the config, Use different **config** to test different models. 

"-o" denotes the output path 

"-t" denotes the test time augmentation (TTA), can be [None, 'lr', 'd4'], default is None, 'lr' is flip TTA, 'd4' is multiscale TTA

"--rgb" denotes whether to output masks in RGB format

**WoodScape**
```
python GeoSeg/vaihingen_test.py -c GeoSeg/config/vaihingen/dcswin.py -o fig_results/vaihingen/dcswin --rgb -t 'd4'
```
