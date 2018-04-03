# A Deep Recurrent Collaborative Filtering Framework for Venue Recommendation (CIKM'17)
In this repository, We're going to implement the paper, which is <b>"A Deep Recurrent Collaborative Filtering Framework for Venue Recommendation"</b>, (Manotumruksa et al, CIKM'17), using a PyTorch library.

## Abstract
Venue recommendation is an important application for Location-Based Social Networks (LBSNs), such as Yelp, and has been extensively
studied in recent years. Matrix Factorisation (MF) is a popular Collaborative Filtering (CF) technique that can suggest relevant
venues to users based on an assumption that similar users are likely to visit similar venues. In recent years, deep neural networks have
been successfully applied to tasks such as speech recognition, computer vision and natural language processing. Building upon this
momentum, various approaches for recommendation have been proposed in the literature to enhance the effectiveness of MF-based
approaches by exploiting neural network models such as: word embeddings to incorporate auxiliary information (e.g. textual content
of comments); and Recurrent Neural Networks (RNN) to capture sequential properties of observed user-venue interactions. However,
such approaches rely on the traditional inner product of the latent factors of users and venues to capture the concept of collaborative
fltering, which may not be sufficient to capture the complex structure of user-venue interactions. In this paper, we propose a
Deep Recurrent Collaborative Filtering framework (DRCF) with a pairwise ranking function that aims to capture user-venue interactions
in a CF manner from sequences of observed feedback by leveraging Multi-Layer Perception and Recurrent Neural Network
architectures. Our proposed framework consists of two components: namely Generalised Recurrent Matrix Factorisation (GRMF)
and Multi-Level Recurrent Perceptron (MLRP) models. In particular, GRMF and MLRP learn to model complex structures of user-venue
interactions using element-wise and dot products as well as the concatenation of latent factors. In addition, we propose a novel
sequence-based negative sampling approach that accounts for the sequential properties of observed feedback and geographical location
of venues to enhance the quality of venue suggestions, as well as alleviate the cold-start users problem. Experiments on three large
checkin and rating datasets show the effectiveness of our proposed framework by outperforming various state-of-the-art approaches.

## Model description
<p align="center">
<img src="/figures/model_description.png" width="700px" height="auto">
</p>

## Getting Started
### Dataset Preparation
The data is represented in the following format:
```bash
<user> \t <venue> \t <timestamp>
```

### Prerequisites
- python 2.7
- tensorflow r1.6

## Notification
We employ static-pool negative sampling technique in this implementation.
