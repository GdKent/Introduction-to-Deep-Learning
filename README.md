# Introduction to Deep learning

![](img1.PNG)

This repository serves as a place where I will keep a portfolio of a variety of different deep learning projects. These projects are meant to serve as references for working with different types of deep learning models on different unstructures data types (images, audio, text, multi-modal, etc.).

## Table of Contents
1. Introduction to Deep Learning
2. Audio-Net: Audio Classification via Image Recognition
3. Austen-Net: Transformer Decoder Language Model for Text Generation
4. Multi-Net: Multi-Modal Architectures
5. Acknowledgments
6. License

## Introduction to Neural Networks with Scikit-Learn and Pytorch
This goal of this project is to serve as a gentle introduction to neural networks (NNs). The code associated with this project is straightforward and is mostly meant to serve as a reference to working with NNs in Scikit-Learn and PyTorch. As such, this project can be split into two parts: (1) NNs with Scikit-Learn and (2) NNs with PyTorch.

The first part of this project starts by implementing a simple multi-layer perceptron (MLP, i.e., a fully-connected feed-forward NN) in Scikit-Learn on a simple tabular Loan Dataset. This consists of importing the dataset, performing simple visualizations and exploratory analysis, followed by training a simple MLPClassifier model in Scikit-Learn. It bears mentioning that none of the NNs in this project are hyper-parameter tuned to perfection, as the project is mainly meant to serve as a reference / introduction to the syntax and process of working with NNs.

The second part of this project focuses on implementing a similar MLP with PyTorch, but now on images (a form of unstructured data). This is accomplished by importing the classic MNIST dataset which consists of images of hand-written digits.

![](Figures/Intro_Figures/MNIST_Batch.PNG)

I then introduce code for going about implementing a simple MLP for this dataset along with the algorithm for training the model. Notice that although MNIST is a dataset of images, I only utilize a feed-forward network to start and not a convolutional NN (CNN), which I will incorporate later. The loss of the model can be plotted over the training iterations (and is displayed below).

![](Figures/Intro_Figures/FCNN_MNIST_Training.PNG)



## Audio-Net: An Audio Classification Model via Image Recognition

The goal of this project is to serve as an introduction to more advanced deep learning techniques that can be applied to solving problems with unstructured data, in this case audio data. 
Specifically, this project aims at training a deep NN to classify different audio samples (a time-dependant sequential data-type) via image recognition. This is accomplished by converting each of the audio wave samples to images of their corresponding mel-scale spectrograms by applying the short-time Fourier-transform (STFT). One can then train a convolutional NN on the resulting spectrogram images, which in this case is accomplished my importing a pre-trained ResNet18 model and fine-tuning it to our dataset via transfer learning.

The dataset that was utilized for this project was the well-known [GTZAN music genre classification dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification?resource=download), which consists audio wave samples of songs / instruments and where the goal is to propperly classify the music genre of any given audio sample (jazz, metal, disco, rock, etc.).

(Insert image here of raw audio-wave and corresponding spectrogram)

The code provided for this project features a series of different data augmentations that can be applied to the spectrogram images to yeild more training data as well as help the model generalize better. Further, the loss of the model can be plotted over the training iterations (as displayed below) and the training algorithm itself is set-up to plot real-time model improvements.

![](Figures/Audio-Net_Figures/Audio-Net_Training.png)

Once the model has been trained, one can devise any number of evaluation metrics to understand the performance of the model on new data. Below is a simple display of some spectrograms, paired with their true labels and the predicted label. Similar to the first project, the model in this project is has by no means been hyper-parameter tuned to perfection, and performance could certainly be improved by choosing more specialized learning parameters as well as implementing different data augmentations. However, as was seen above, the model is indeed able to learn, which is the only goal of the remaining projects (to have successfully implementable code for training these models; past that, it is a matter of hyper-parameter tuning and throwing more GPUs and training time at it...).

![](Figures/Audio-Net_Figures/Audio-Net_Batch_Accuracy.png)



## Austen-Net: A Transformer Decoder Language Model for Text Generation

Large-Language models have take the world by storm ever since OpenAI's ChatGPT (GPT standing for Generative Pre-Trained Transformer) went mainstream in 2022; however, the deep NN architectures under the hood of these models (Transformers) immediately began taking over many diffferent problem domains ever since the release of the seminal paper [Attention is All you Need](https://arxiv.org/abs/1706.03762) in 2017, which proposed the simple-yet-effective idea of the attention operator as well as the general transformer architecture. From the original transformer architecture that was proposed in that paper (which would would now be referred to as an encoder-decoder transformer), several sub-architectures were found to be useful for accomplishing different tasks: (1) the encoder-only transformer and (2) the decoder-only transformer. Encoder-only transformers like BERT (Bidirectional Encoder Representations from Transformers) and its variants (HuBERT, ALBERT, etc.) excel at typical supervised learning classification / regression problems. Alternatively, decoder-only transformers like ChatGPT (Google's Gemini, etc.) excel at next-token prediction tasks for sequential data (earning them the name of "Generative Models" as they are able to "generate" the next token in a sequence). Lastly, Encoder-Decoder transformers like Meta's BART (or Google's T5) excel at sequence-conversion tasks, like translating engligh to spanish, etc.

The goal of this project is to serve as an introduction to utilizing these transformer architectures on a language modeling task. Specifically, the project aims at training a customly-implemented decoder-only transformer to generate infinite text in the style of Jane Austen (hence the name Austen-Net). This was done by training the model on a corpus of Austen's primary works. Specifically, the dataset consisted of the eight following novels by Jane Austen and were obtained from [Project Gutenberg](https://www.gutenberg.org/):
- Persuasion
- Northanger Abbey
- Mansfield Park
- Emma
- Lady Susan
- Love and Friendship and Other Early Works
- Pride and Prejudice
- Sense and Sensibility

The loss of the model can be plotted over the training iterations (as displayed below) and the training algorithm itself is set-up to plot real-time model improvements.

![](Figures/Austen-Net_Figures/Austen-Net_Training.png)

Lastly, one can evaluate the before-and-after training performance of the model by observing the text that is generated by the model given some input seed text. Notice that the code currently only returns a sequence of generated tokens up to the maximum context-length that it was trianed with. However, it can trivially be altered to feed its predicted output back into it to generate infinite amounts of text.

![](Figures/Austen-Net_Figures/Generated_Text.PNG)





## Multi-Net: Multi-Modal Architecture for Video Recommendation

The goal of this project is to serve as a reference for implementing multi-modal archuitectures to solve complex deep learning problems. Specifically, this project consists of building a video recommendation engine using the well-known [MSR-VTT Dataset](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/06/cvpr16.msr-vtt.tmei_-1.pdf) (downloadable at [The Cove](https://cove.thecvf.com/datasets/839)). As such, this dataset consists of three different modalities: video, audio, and text. This is accomplished by generating embeddings (where each embedding comes from a pre-trained model that is fine-tuned to this dataset via transfer learning) for each of the modalities and combining them in an early fusion design. The model is then trained in a representation learning manner with a contrastive loss function to pull similar embeddings closer together in the resulting vector space while pushing dissimilar embeddings further apart. Below is a high-level overview of each of these components as well as their implementations.

### Video Encoder (ResNet18)

To generate embeddings related to the video frame data, we utilize the following methodology (see the "Video Encoder" visualization displayed below):
- (1) Utilize the PyAV library to gather the image frames from the video.
- (2) Sample frames from the video - this will help keep the computational cost down while still ideally representing the visual information present throughout the video.
- (3) Transform and apply augmentations to each of the sample frames.
- (4) Compute the corresponding embeddings for each of the sampled frames by feeding them into a pretrained ResNet18 model (see the paper [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)).
- (5) Aggregate the sample frame embeddings by averaging them together, yielding a single embedding that describes the visual content of the video.
- (6) Project the aggregated embedding into a final N-dimensional vector space. This final embedding will be the video embedding.

(Image Here)

### Audio Encoder (HuBERT)

To generate embeddings related to the audio data, we utilize the following methodology (see the "Audio Encoder" visualization displayed below):
- (1) Utilize the PyAV library to gather the raw audio data from the video.
- (2) Transform the audio data into usable features for HuBERT by utilizing its feature extractor (this is essentially a Wave2VecFeatureExtractor under the hood which is specialized for HuBERT).
- (3) Compute the embedding of the audio wave by feeding its features into a pretrained HuBERT model (see the paper [HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units](https://arxiv.org/abs/2106.07447)).
- (4) Project the resulting embedding into a final N-dimensional vector space. This final embedding will be the audio embedding.


### Text Encoder (ALBERT)

To generate embeddings related to the text data, we utilize the following methodology (see the "Text Encoder" visualization displayed below):
- (1) Collect all of the captions into a list that correspond to each video (about 20 captions per video).
- (2) Sample some number of the captions for a video.
- (3) Tokenize each sampled caption by utilizing the ALBERT tokenizer.
- (4) Compute the corresponding embeddings for each of the sampled tokenized captions by feeding them into a pretrained ALBERT model (see the paper [ALBERT: A Lite BERT for Self-supervised Learning of Language Representations](https://arxiv.org/abs/1909.11942)).
- (5) Aggregate the sampled caption embeddings by averaging them together, yielding a single embedding that summarizes the textual content of the video.
- (6) Project the aggregated embedding into a final N-dimensional vector space. This final embedding will be the text embedding.

### Early Fusion Multi-Modal Architecture

![](Figures/Multi-Net_Figures/Multi-Net_Training.png)



## Acknowledgements



## License







