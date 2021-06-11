# Few-Shot-Classification
The objective of this project is to evaluate the feasibility of using pre-trained feature-extractors to quickly categorize products based on images with limited amounts of data. To do that, we explore different techniques based on metric learning (siamese and prototypical networks) and meta-learning (model-agnostic meta-learning).

## Siamese Networks
They consist of neural networks that contain two or more identical sub-networks, that share same characteristics and parameters and undergo the same updates during training.
![img](https://github.com/carlossantosgarcia/few-shot-classification/blob/main/images/siamese_networks_diagram.png)