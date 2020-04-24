# Temporally Coherent Embeddings for Self-Supervised Video Representation Learning
This repository contains the code implementation used in the paper Temporally Coherent Embeddings for Self-Supervised Video Representation Learning (TCE). \[[arXiv](https://arxiv.org/abs/2004.02753)] \[[Website](https://csiro-robotics.github.io/TCE_Webpage/)]  Our contributions in this repository are:
- A Pytorch implementation of the self-supervised training used in the TCE paper
- Pre-trained checkpoints for models trained using the TCE self-supervised training paradigm
- A Pytorch implementation of action-recognition fine-tuning on the UCF101 dataset using a stack-of-differences frame encoder for a given pre-trained checkpoint
- A Pytorch implementation of t-SNE visualisations of the network output, which can be repurposed across any number of network architectures

![Network Architecture](images/TCE.png)

We benchmark our code on Split 1 of the UCF101 action recognition dataset, providing pre-trained models for our downstream and upstream training.  See [Models](#models) for our provided models and Getting Started (LINK) for for instructions on training and evaluation.

If you find this repo useful for your research, please consider citing the paper
 ```
@misc{knights2020temporally,
    title={Temporally Coherent Embeddings for Self-Supervised Video Representation Learning},
    author={Joshua Knights and Anthony Vanderkop and Daniel Ward and Olivia Mackenzie-Ross and Peyman Moghadam},
    year={2020},
    eprint={2004.02753},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
 ```



## Updates
- 23/04/2020 : Initial Commit



## Table of Contents

- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Models](#models)
- [Getting Started](#getting-started)
- [Acknowledgements](#acknowledgements)

<a name="installation"></a>
## Installation

TCE is built using Python = 2.7 and pytorch = 0.4.1

- Setup Python environment using conda:
```
conda env create -f TCE_env.yml
conda activate env
```
- Add the project to your PYTHONPATH
```
export PYTHONPATH=$PWD:$PYTHONPATH
```
<a name="data-preparation"></a>



## Data Preparation

UCF101 frames can be downloaded directly from [feichtenhofer/twostreamfusion](https://github.com/feichtenhofer/twostreamfusion)
```
wget http://ftp.tugraz.at/pub/feichtenhofer/tsfusion/data/ucf101_jpegs_256.zip.001
wget http://ftp.tugraz.at/pub/feichtenhofer/tsfusion/data/ucf101_jpegs_256.zip.002
wget http://ftp.tugraz.at/pub/feichtenhofer/tsfusion/data/ucf101_jpegs_256.zip.003

cat ucf101_jpegs_256.zip* > ucf101_jpegs_256.zip
unzip ucf101_jpegs_256.zip
```
Lists containing the various train/test splits for UCF101 can be found in UCF_list.  In finetune.py, replace the values for `__UCF101DIREC__` and `__UCF101SPLITS__` with paths to your UCF101 images and split folders.



<a name="models"></a>

## Models
| Architecture | Dataset / Split | Self-Supervised Checkpoint                                                                     | Action Recognition Checkpoint                                                                  | Accuracy |
|--------------|-----------------|------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------|----------|
| ResNet-50    | UCF101 Split 1  | [Download](https://cloudstor.aarnet.edu.au/plus/s/D5o3Ip8PQrZSFpM) | [Download](https://cloudstor.aarnet.edu.au/plus/s/0a4H0yfFGpLK0wr) | 66.67    |
| ResNet-101   | UCF101 Split 1  | [Download](https://cloudstor.aarnet.edu.au/plus/s/cprZhEwvSbAg2mV) | [Download](https://cloudstor.aarnet.edu.au/plus/s/Qb0u9f0rMoEaz9K) | 68.7     |


## Getting Started

### Self-Supervised Training
Code for self-supervised pretraining using TCE will be released at a future date. 

### Fine-tuning for action recognition
We provide `finetune.py` for fine-tuning the network on the UCF101 action recognition dataset from an existing checkpoint, such as imagenet pre-training or our self-supervised training.

#### Flags
The following flags can be used to adjust the configuration of the fine-tuning:
- `--weights` : Path to initial pre-trained weights for fine-tuning
- `--resume` : Path to partially fine-tuned checkpoint to resume from
- `--savedir` : Path to save models and tensorboards to
- `--model` : Model to use for the network backbone.  Current options are resnet18, 34, 50, 101, and 150.
- `--split` : Choice of train/test split 01, 02, or 03.
- `--evaluate` : Only run evaluation on a given checkpoint

Additional flags can be found by looking inside `finetune.py`

#### Inference demo with finetuned model
Run the finetune script in evaluation mode with the following command:
```
python finetune.py --evaluate --model resnet50 --split 01 --resume /path/to/checkpoint.pth --savedir /path/to/savedir 
```
#### Fine-tuning from a pre-trained checkpoint
Run the finetune script in training mode with the following command:
```
python finetune.py --model resnet50 --split 01 --weights /path/to/weights.pth --savedir /path/to/savedir 
```



### Visualisation
![vid](images/video_tsne_example.gif)

We provide along with the trained models a package to create t-SNE visualisations similar to those found in the paper.  Our t-SNE code is capable of performing the following:
- Demonstrating the temporal coherency of our embedding space by reducing the dimensionality of our embeddings to 2D for plotting
- Plotting multiple input videos alongside each other
- Plotting an input video alongside the video it is representing

In addition, our code can be used to help create t-SNE visualisations for checkpoints and architectures other than that used in the TCE paper.  

The scripts in this repository used for generating t-SNE visualisations are:
- `make_tsne.py` Is a wrapper for t-SNE and our network architecture for end-to-end generation of the t-SNE
- `tsne_utils.py` Contains t-SNE functionality for reducing the dimensionality of an array of embedded features for plotting
- `make_tsne_from_embedding.py` Creates a t-SNE plot from a numpy array containing the embeddings of a single video

#### Flags
The following flags can be used as inputs for `make_tsne.py`:
- `--model` : Model to use for the network backbone.  Current options are resnet18, 34, 50, 101, and 150.
- `--input` : Input to be plotted.  Accepts as input both video files and folders of frames.  This flag can take multiple videos as input, seperated by spaces.
- `--vid` : If set, saves the t-SNE as an MP4 video that demonstrates the progression of the embeddings through the featurespace alongside the input video.  Only applicable when the number of input videos is one.
- `--output` : Path to save network output to.
- `--ckpt` : Checkpoint containing weights for the network

#### Visualising TCE
To visualise the embeddings from TCE, download our self-supervised model above and use the following command to visualise our embedding space as a video:
```
python make_tsne.py --vid --model resnet50 --input /path/to/input --output /path/to/output.mp4 --ckpt /path/to/TCE_ckpt.pth
```
And the following command to visualise in a PNG image:
```
python make_tsne.py --model resnet50 --input /path/to/input --output /path/to/output.mp4 --ckpt /path/to/TCE_ckpt.pth
```
Examples of t-SNEs that can be generated using our repository can be found in images/

Our visualisation package can also be used to create t-SNE visualisations of checkpoints from other network architectures.  To do so:
1. Generate and save a numpy array containing frame embeddings for your video as an [N, D] dimensional array where N is the number of frames in the video and D is the dimensionality of the embeddings.  These embeddings should be chronologically sorted from start to finish across the 0th axis of the array
2. Generate the t-SNE using the following command:
```
python make_tsne_from_embedding.py --embeddings /path/to/embeddings --output --path/to/output.png
```
An example of a numpy array of embedded features that can be used to generate a t-SNE this way can be downloaded from [here](https://drive.google.com/file/d/1UAvgiRU67DCII653BU8F51oZpXsuHoa2/view?usp=sharing).  Alternatively, you can write your own script to wrap your network architecture with the t-SNE and plotting functions used in this repository.

<a name="acknowledgements"></a>



## Acknowledgements
Part of this code is inspired by Yonglong Tian's unsupervised learning algorithm [Contrastive Multiview Coding](https://github.com/HobbitLong/CMC) and Jeffrey Huang's implementation of [action recognition](https://github.com/jeffreyyihuang/two-stream-action-recognition).




