# Few-Shot-Classification (FSL)
The objective of this project is to evaluate the feasibility of using pre-trained feature-extractors to quickly categorize products based on images with limited amounts of data. To do that, we explore different techniques based on metric learning (siamese and prototypical networks) and meta-learning (model-agnostic meta-learning). Our results are presented in our [defense presentation](https://github.com/carlossantosgarcia/few-shot-classification/blob/main/project%20defense/presentation.pdf) and [report](https://github.com/carlossantosgarcia/few-shot-classification/blob/main/project%20defense/report.pdf).

## Dataset
We use real images from seven luxury brands. The dataset provided by Navee contains 3967 classes accross 7 brands. Each class represents a fashion article and contains about 5 images. Here is an example of the images available for three different articles.
![img](https://github.com/carlossantosgarcia/few-shot-classification/blob/main/images/example.png)

## Image Retrieval task 
We approach FSL through a retrieval task that is evaluated with mean average precision (mAP): our systems embed images in a space where similar articles should be projected closer to one another. During training, mAP metrics are computed to check on the performances of our networks.
![img](https://github.com/carlossantosgarcia/few-shot-classification/blob/main/images/retrieval.png)

## Siamese Networks
They consist of neural networks that contain two or more identical sub-networks, that share same characteristics and parameters and undergo the same updates during training.
![img](https://github.com/carlossantosgarcia/few-shot-classification/blob/main/images/siamese_networks_diagram.png)
Two main losses have been used to train our models: contrastive loss [[1]](#1) and triplet loss, especially developed in [[2]](#2). The former is based on using pairs of images. The latter is based on the use of triplets of images. The idea of both losses is to **push similar images close together** and **dissimilar images far from another** in the embedding space. 

## Prototypical Networks
Prototypical networks are a metric-learning technique using in our implementation ResNet-50 to map fashion images into a metric space where classification is then done computing prototypes (means) from each category and their distance to the query image. This simple method can actually thrive in limited-data regime.

## Model-Agnostic Meta-Learning
This is an implementation of the paper by Finn et. al[[3]](#3), that uses meta-learning to train a model on batches of tasks, for the purpose of image classification using few-shot 
learning. Although it's model-agnostic, since it can be implemented with any gradient-descent model, we use a convolutional network. Training is composed of training on each
individual task then minimizing the sum of all losses. This implementation is heavily inspired by [this one](https://github.com/dragen1860/MAML-Pytorch).

## References
<a id="1">[1]</a> 
LeCun et al. (2005). 
_Dimensionality reduction by learning an invariant mapping_

<a id="2">[2]</a> 
Schroff et al. (2015). 
_FaceNet: A Unified Embedding for Face Recognition and Clustering_

<a id="3">[3]</a> 
Finn et al. (2017). 
_Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks_
