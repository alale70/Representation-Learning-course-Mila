# Variational Auto-encoder and GAN on MNIST Dataset

### Representational Learning Course Project  
**Professor:** Aaron Courville  
**Institution:** Université de Montréal (UdeM)  

---

## Project Overview

This project was completed as part of the Deep Learning course at Polytechnique Montréal. The aim was to develop and evaluate two advanced generative models on the MNIST dataset: a Variational Auto-encoder (VAE) and a Generative Adversarial Network (GAN). 

---

## Objectives

1. **Variational Auto-encoder (VAE)**:
   - **Architecture**: Defined and developed a VAE model for the MNIST dataset.
   - **Training and Evaluation**: Conducted model training and evaluation. Utilized importance sampling to enhance model performance and accuracy.

2. **Generative Adversarial Network (GAN)**:
   - **Implementation**: Implemented and trained a GAN for image generation using the MNIST dataset.
   - **Evaluation**: Compared the quality and performance of the generated images with the original dataset to assess model effectiveness.

---

## Datasets

- **MNIST Dataset**: A dataset of handwritten digits, consisting of 60,000 training images and 10,000 test images, each 28x28 pixels in size.

---

## Models

### Variational Auto-encoder (VAE)

- **Objective**: Develop a VAE to learn a probabilistic model of the MNIST data.
- **Architecture**: The model includes an encoder that maps input data to a latent space and a decoder that reconstructs the data from the latent representation.
- **Techniques**: Importance sampling was utilized to improve performance.

### Generative Adversarial Network (GAN)

- **Objective**: Train a GAN to generate new images similar to those in the MNIST dataset.
- **Architecture**: Consists of a generator network that creates synthetic images and a discriminator network that distinguishes between real and generated images.
- **Evaluation**: The quality of generated images was compared to the original dataset to gauge model effectiveness.

