# Machine Learning & Deep Learning Course
Homeworks from the Machine Learning and Deep Learning master course at Polythecnic of Turin.

## Homeworks
### HM1
Study the [wine dataset](https://archive.ics.uci.edu/ml/datasets/wine) and use different classification techniques to label each element to the right class.  
Algorithms used: Nearest Neighbors, Support Vector Machine with Linear and RBF kernerl. 

### HM2
Train a Convolution Neural Network for image classification using the [Caltech-101 dataset](http://www.vision.caltech.edu/Image_Datasets/Caltech101/).  
Neural networks were trained using 2 techniques: training from scratch and using **transfer learning**, with the Alexnet pre-trained NN.  
Extra tests were also performed with VGG16 and RESNET18 NNs as backbones.

### HM3
The task is to implement DANN, a Domain Adaptation algorithm, on the [PACS dataset](https://paperswithcode.com/dataset/pacs) using AlexNet NN.
PACS is an image dataset for domain generalization. It consists of four domains, namely Photo (1,670 images), Art Painting (2,048 images), Cartoon (2,344 images) and Sketch (3,929 images). Each domain contains seven classes: dog, elephant, giraffe, guitar, horse, house, person.  
Tests were performed with normal training and domain adaptation training.  
Extra tests were performed splitting the database in the different domains:  train on the Photo dataset evaluate on Cartoon dataset, train it again on Photo dataset and evaluate it on Sketch dataset.

## Results
Each HW was graded to up to 2 point. I scored perfectly getting 6 points out of 6.
