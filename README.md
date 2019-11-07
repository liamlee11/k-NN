# Implementation of k-NN Classifier

## Goals and Overview

***Goals and Overview:*** You will be assigned to implement k-nearest neighbors (k-NN) classifier, a type of linear classifier, to understand the basic concept. You need to make your own script containing train and test code for k-NN classifier. You can use the function loading CIFAR-10 dataset. You have to implement the code changing k.

## Summary

***Hyperparameter tuning by Cross-validation***

![graph](/graph.png)

Hyperparameter tuning 을 위한 5-Fold cross-validation 모델을 그래프로 구조화하였다.

전체 Train data 는 같은 크기를 가진 5개의 dataset 으로 랜덤하게 분배되고(로드된 5개의 데이터셋이 랜덤하게 분배되어 있다고 가정) 각 dataset 을 node 로 하는 directed graph 를 만들 수 있다. 이때 연결되는 edge 는 20개인데 Edge(ij) 는 node(i) 를 train data 로 하고 node(j) 를 test data 로 하는 것으로 정의한다. 따라서 Edge(ij) = transpose(Edge(ji)) 가 성립하게 되므로 결과적으로 10개의 Edge connection 의 조합으로 5-fold cross validation 을 수행할 수 있다.

이러한 그래프 모델을 가져온 이유는 알고리즘의 시간 복잡도를 최소화하기 위함이고, 10개의 Edge 조합에 의한 데이터셋은 가능한 최소한의 계산을 요구한다. 예를 들어 5개 중 node1을 test data 로 사용하는 fold dataset 의 컨셉은 다음과 같을 것이다. Fold1 = array(Edge21, Edg31, Edg41, Edg51) = array(Edge12.T, Edge13.T, Edge14.T, Edge15.T) (T는 전치행렬)
