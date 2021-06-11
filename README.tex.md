# Few-Shot-Classification (FSL)
The objective of this project is to evaluate the feasibility of using pre-trained feature-extractors to quickly categorize products based on images with limited amounts of data. To do that, we explore different techniques based on metric learning (siamese and prototypical networks) and meta-learning (model-agnostic meta-learning).

## Dataset
We use real images from seven luxury brands. The dataset provided by Navee contains 3967 classes accross 7 brands. Each class represents a fashion article and contains about 5 images. Here is an example of the images available for three different articles.
![img](https://github.com/carlossantosgarcia/few-shot-classification/blob/main/images/example.png)

## Image Retrieval task 
We approach FSL through a retrieval task that is evaluated with mean average precision (mAP): our systems embed images in a space where similar articles should be projected closer to one another. During training, mAP metrics are computed to check on the performances of our networks.
![img](https://github.com/carlossantosgarcia/few-shot-classification/blob/main/images/retrieval.png)

## Siamese Networks
They consist of neural networks that contain two or more identical sub-networks, that share same characteristics and parameters and undergo the same updates during training.
![img](https://github.com/carlossantosgarcia/few-shot-classification/blob/main/images/siamese_networks_diagram.png)
Two main losses have been used to train our models: contrastive loss and triplet loss. The former is based on using pairs of images and can be expressed as: $$L_{contrastive}=y\times d(x,x')+(1-y)\times\max(0,m-d(x,x'))$$
where $x$ and $x'$ are the images fed into the network,  forming a positive pair (same class : $y=1$) or negative pair (different classes : $y=0$).
The lattes is based on the use of triplets of images: an anchor sample $x^a$, a positive sample $x^p$ (same class as the anchor) and a negative sample $x^n$ (different class). This loss can be written as:
$$ L_{triplet}=max\big(0,d(x^a,x^p)+m-d(x^a,x^n)\big)$$

The idea is to **push similar images close together** and **dissimilar images far from another** in the embedding space. 

## Prototypical Networks

## Model-Agnostic Meta-Learning