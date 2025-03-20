# Introduction to Deep learning

![](img1.PNG)

This repository serves as a place where I will keep a portfolio of a variety of different deep learning projects. These projects are meant to serve as references for working with different types of deep learning models on different unstructures data types (images, audio, text, multi-modal, etc.).

## Table of Contents
1. [Introduction to Deep Learning](#intro)
2. [Audio-Net: Audio Classification via Image Recognition](#audio-net)
3. [Austen-Net: Transformer Decoder Language Model for Text Generation](#austen-net)
4. [Multi-Net: Multi-Modal Architectures](#multi-net)
5. [Acknowledgments](#acknowledgments)
6. [License](#license)

## Introduction to Neural Networks with Scikit-Learn and Pytorch
This goal of this project is to serve as a gentle introduction to neural networks (NNs). The code associated with this project is straightforward and is mostly meant to serve as a reference to working with NNs in Scikit-Learn and PyTorch. As such, this project can be split into two parts: (1) NNs with Scikit-Learn and (2) NNs with PyTorch.

The first part of this project starts by implementing a simple multi-layer perceptron (MLP, i.e., a fully-connected feed-forward NN) in Scikit-Learn on a simple tabular Loan Dataset. This consists of importing the dataset, performing simple visualizations and exploratory analysis, followed by training a simple MLPClassifier model in Scikit-Learn. It bears mentioning that none of the NNs in this project are hyper-parameter tuned to perfection, as the project is mainly meant to serve as a reference / introduction to the syntax and process of working with NNs.

The second part of this project focuses on implementing a similar MLP with PyTorch, but now on images (a form of unstructured data). This is accomplished by importing the classic MNIST dataset which consists of images of hand-written digits.

![](Figures/Intro_Figures/MNIST_Batch.PNG)

I then introduce code for going about implementing a simple MLP for this dataset along with the algorithm for training the model. Notice that although MNIST is a dataset of images, I only utilize a feed-forward network to start and not a convolutional NN (CNN), which I will incorporate later. The loss of the model can be plotted over the training iterations (and is displayed below).

![](Figures/Intro_Figures/FCNN_MNIST_Training.PNG)



## Audio-Net: An Audio Classification Model via Image Recognition

The dataset that was utilized for this project was the well-known [GTZAN music genre classification dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification?resource=download).

![](Figures/Audio-Net_Figures/Audio-Net_Training.png)

![](Figures/Audio-Net_Figures/Audio-Net_Batch_Accuracy.png)



## Austen-Net: A Transformer Decoder Language Model for Text Generation

The dataset consisted of the eight following novels by Jane Austen:
- Persuasion
- Northanger Abbey
- Mansfield Park
- Emma
- Lady Susan
- Love and Friendship and Other Early Works
- Pride and Prejudice
- Sense and Sensibility

![](Figures/Austen-Net_Figures/Austen-Net_Training.png)

![](Figures/Austen-Net_Figures/Generated_Text.PNG)










## Multi-Net: Multi-Modal Architecture for Video Recommendation
This project consists of building a video recommendation engine using the well-known [MSR-VTT Dataset](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/06/cvpr16.msr-vtt.tmei_-1.pdf). This is accomplished by generating embeddings (where each embedding comes from a pre-trained model that is fine-tuned to this dataset via transfer learning) for each of the modalities and combining them in an early fusion design. The model is then trained via representation learning with a contrastive loss function to pull similar embeddings closer together in the resulting vector space while pushing dissimilar embeddings further apart.

### Video Encoder


### Audio Encoder


### Text Encoder


### Early Fusion Multi-Modal Architecture

![](Figures/Multi-Net_Figures/Multi-Net_Training.png)



## Acknowledgements



## License







