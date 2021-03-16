# Temporally Coherent Embeddings for Self-Supervised Video Representation Learning
This repository contains the code implementation used in the ICPR2020 paper Temporally Coherent Embeddings for Self-Supervised Video Representation Learning (TCE). \[[arXiv](https://arxiv.org/abs/2004.02753)] \[[Website](https://csiro-robotics.github.io/TCE-Webpage/)]  Our contributions in this repository are:
- A Pytorch implementation of the self-supervised training used in the TCE paper
- A Pytorch implementation of action recognition fine-tuning
- Pre-trained checkpoints for models trained using the TCE self-supervised training paradigm
- A Pytorch implementation of t-SNE visualisations of the network output

![Network Architecture](images/TCE.png)

We benchmark our code on Split 1 of the UCF101 action recognition dataset, providing pre-trained models for our downstream and upstream training.  See [Models](#models) for our provided models and Getting Started (#getting-started) for for instructions on training and evaluation.

If you find this repo useful for your research, please consider citing the paper
 ```
@inproceedings{knights2020tce,
  title={Temporally Coherent Embeddings for Self-Supervised Video Representation Learning},
  author={Joshua Knights and Ben Harwood and Daniel Ward and Anthony Vanderkop and Olivia Mackenzie-Ross and Peyman Moghadam},
 booktitle={25th International Conference on Pattern Recognition (ICPR)},
  year={2020}
}

 ```


## Updates
- 23/04/2020 : Initial Commit
- 30/11/2020 : ICPR Update

## Table of Contents

- [Data Preparation](#data-preparation)
- [Installation](#installation)
- [Models](#models)
- [Getting Started](#getting-started)
- [Acknowledgements](#acknowledgements)


## Data Preparation 
<a name="data-preparation"></a>

### Kinetics400
Kinetics400 videos can be downloaded and split into frames directly from [Showmax/kinetics-downloader](https://github.com/Showmax/kinetics-downloader)

The file directory should have the following layout:
```
├── kinetics400/train
    |
    ├── CLASS_001
    ├── CLASS_002
    .
    .
    .
    CLASS_400
        | 
        ├── VID_001
        ├── VID_002
        .
        .
        .
        ├── VID_###
            | 
            ├── frame1.jpg
            ├── frame2.jpg
            .
            .
            .
            ├── frame###.jpg
```
Once the dataset is downloaded and split into frames, edit the following parameters in config/default.py to point towards the frames and splits:
- DATASET.KINETICS400.FRAMES_PATH = /path/to/kinetics400/train

### UCF101

UCF101 frames and splits can be downloaded directly from [feichtenhofer/twostreamfusion](https://github.com/feichtenhofer/twostreamfusion)

```
wget http://ftp.tugraz.at/pub/feichtenhofer/tsfusion/data/ucf101_jpegs_256.zip.001
wget http://ftp.tugraz.at/pub/feichtenhofer/tsfusion/data/ucf101_jpegs_256.zip.002
wget http://ftp.tugraz.at/pub/feichtenhofer/tsfusion/data/ucf101_jpegs_256.zip.003

cat ucf101_jpegs_256.zip* > ucf101_jpegs_256.zip
unzip ucf101_jpegs_256.zip
```
The file directory should have the following layout:

```
├── UCF101
    |
    ├── v_{_CLASS_001}_g01_c01
    .   | 
    .   ├── frame000001.jpg
    .   ├── frame000002.jpg 
    .   .
    .   .
    .   ├── frame000###.jpg
    .
    ├── v_{_CLASS_101}_g##_c##
        | 
        ├── frame000001.jpg
        ├── frame000002.jpg 
        .
        .
        ├── frame000###.jpg
```

Once the dataset is downloaded and decompressed, edit the following parameters in config/default.py to point towards the frames and splits:
- DATASET.UCF101.FRAMES_PATH = /path/to/UCF101_frames
- DATASET.UCF101.SPLITS_PATH = /path/to/UCF101_splits





## Installation
<a name="installation"></a>

TCE is built using Python == 3.7.1 and PyTorch == 1.7.0

We use Conda to setup the Python environment for this repository.  In order to create the environment, run the following commands from the root directory:

```
conda env create -f TCE.yaml
conda activate TCE
```

Once this is done, also specify a path to save assets (such as dataset pickles for faster setup) to in config.default.py:
- ASSETS_PATH = /path/to/assets/folder



## Models
<a name="models"></a>

| Architecture 	| Pre-Training Dataset 	| Link                                                           	|
|--------------	|----------------------	|----------------------------------------------------------------	|
| ResNet-18    	| Kinetics400          	| [Link](https://cloudstor.aarnet.edu.au/plus/s/kNQKw5ATTbyamg2) 	|
| ResNet-50    	| Kinetics400          	| [Link](https://cloudstor.aarnet.edu.au/plus/s/HbWxmhcUbfzQIQf) 	|

## Getting Started

### Self-Supervised Training
We provide a script for pre-training with the Kinetics400 dataset using TCE, pretrain.py.  To train, run the following script:

```
python finetune.py \
    --cfg config/pretrain_kinetics400miningr_finetune_UCF101_resnet18.yaml  \
    TRAIN.PRETRAINING.SAVEDIR /path/to/savedir 
```

If resuming from a previous pre-training checkpoint, set the flag `TRAIN.PRETRAINING.CHECKPOINT` to the path to the checkpoint to resume from

### Fine-tuning for action recognition
We provide a fine-tuning script for action recognition on the UCF-101 dataset, finetune.py.  To train, run the following script:

```
python finetune.py \
    --cfg config/pretrain_kinetics400miningr_finetune_UCF101_resnet18.yaml \
    TRAIN.FINETUNING.CHECKPOINT "/path/to/pretrained_checkpoint" \
    TRAIN.FINETUNING.SAVEDIR "/path/to/savedir"
```

If resuming training from an earlier finetuning checkpoint, set the flag `TRAIN.FINETUNING.RESUME` to True 




### Visualisation

![vid](images/bowling_tsne_example.gif)

In order to demonstrate the ability of our approach to create temporally coherent embeddings, we provide a package to create t-SNE visualisations of our features similar to those found in the paper.  This package can also be applied to other approaches and network architectures.

The files in this repository used for generating t-SNE visualisations are:
- `visualise_tsne.py` Is a wrapper for t-SNE and our network architecture for end-to-end generation of the t-SNE
- `utils/tsne_utils.py` Contains t-SNE functionality for reducing the dimensionality of an array of embedded features for plotting, as well as tools to create an animated visualisation of the embedding's behaviour over time

The following flags can be used as inputs for `make_tsne.py`:
- `--cfg` : Path to config file
- `--target` : Path to video to visualise t-SNE for.  This video can either be a video file (avi, mp4) or a directory of images representing frames
- `--ckpt` : Path to the model chekpoint to visualise the embedding space for
- `--gif` : Use to visualise the change in the embedding space over time alongside the input video as a gif file
- `--fps` : Set the framerate of the gif
- `--save` : Path to save the output t-SNE to

To visualise the embeddings from TCE, download our self-supervised model above and use the following command to visualise our embedding space as a gif:

```
python visualise_tsne.py
    --cfg config/pretrain_kinetics400miningr_finetune_UCF101_resnet18.yaml \
    --target "/path/to/target/video" \
    --ckpt "/path/to/TCE_checkpoint" \
    --gif \
    --fps 25 \
    --save "/path/to/save/folder/t-SNE.gif"
```

Alternatively, to visualise the t-SNE as a PNG image use the following:

```
python visualise_tsne.py
    --cfg config/pretrain_kinetics400miningr_finetune_UCF101_resnet18.yaml \
    --target "/path/to/target/video" \
    --ckpt "/path/to/TCE_checkpoint" \
    --save "/path/to/save/folder/t-SNE.png"
```

<a name="acknowledgements"></a>



## Acknowledgements
Parts of this code base are derived from Yonglong Tian's unsupervised learning algorithm [Contrastive Multiview Coding](https://github.com/HobbitLong/CMC) and Jeffrey Huang's implementation of [action recognition](https://github.com/jeffreyyihuang/two-stream-action-recognition).




