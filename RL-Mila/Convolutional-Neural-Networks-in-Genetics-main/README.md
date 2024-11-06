# DNA Accessibility Classification Using Convolutional Neural Networks

### Representational Learning Course Project  
**Professor:** Aaron Courville  
**Institution:** Université de Montréal (UdeM)  
**Project:** Convolutional Neural Networks for Biological Data

---

## Project Overview

This project was completed as part of the Representational Learning course at Université de Montréal under the supervision of Professor Aaron Courville. The goal of the project is to implement a Convolutional Neural Network (CNN) to classify DNA regions based on accessibility using the **Basset** dataset.

---

## Background

DNA (Deoxyribonucleic Acid) is a molecule that carries genetic instructions for the development, functioning, growth, and reproduction of all living organisms. The DNA in humans can be viewed as a long string of characters {A, C, T, G}, representing nucleotides. Approximately 98% of DNA is physically inaccessible to external molecules, and understanding which regions are accessible can provide valuable insights in biology.  

This project focuses on classifying DNA subsequences from the reference human genome HG19 based on their accessibility, leveraging deep learning techniques. HG19 can be thought of as the DNA of a prototypical human, and the input data for this project will be one-hot encoded DNA sequences.

---

## Dataset

The **Basset** dataset contains subsequences of DNA derived from the human genome HG19. Each DNA sequence is one-hot encoded such that:

$$
A = \begin{pmatrix} 1 \\ 0 \\ 0 \\ 0 \end{pmatrix}, C = \begin{pmatrix} 0 \\ 1 \\ 0 \\ 0 \end{pmatrix}, G = \begin{pmatrix} 0 \\ 0 \\ 1 \\ 0 \end{pmatrix}, T = \begin{pmatrix} 0 \\ 0 \\ 0 \\ 1 \end{pmatrix}
$$

These encoded base pairs (A, C, G, T) are concatenated along the length of the DNA sequence. For data processing, each data point can be considered as a rectangular shape of size (sequence length, 4) with a single channel.

### Target Data

The target data for each sequence is a binary vector representing accessibility. The vector is of size 164, where a value of 1 at an index indicates that the DNA sequence is accessible in that experiment, and 0 otherwise. Our CNN will be trained to output binary multi-label predictions (accessible/inaccessible) based on these target vectors.

### Data Inspection

Before training the model, we inspect the data to understand its structure and ensure it is suitable as an input for machine learning algorithms. Each input sequence represents DNA regions with accessibility labels, and we use CNNs to identify patterns and motifs in the sequences that are correlated with accessibility.

---

## Model

The model architecture follows a Convolutional Neural Network (CNN), designed to extract meaningful patterns from the DNA sequences.

1. **Input Layer**: One-hot encoded DNA sequences.
2. **Convolutional Layers**: These layers capture local patterns in the DNA sequences (e.g., motifs). We expect the first layer filters to learn biologically significant motifs.
3. **Pooling Layers**: These layers reduce the dimensionality of the feature maps while retaining important information.
4. **Fully Connected Layers**: These layers classify the sequences based on the features extracted by the convolutional layers.
5. **Output Layer**: Binary classification predicting whether a DNA sequence is accessible or not.

---

## Objective

The main objective is to predict DNA accessibility using CNNs trained on the Basset dataset. We aim to replicate part of Figure 3b from the original **Basset** paper by Kelley et al. (2016). Specifically, we will train the CNN to identify DNA motifs and assess the model's performance in predicting DNA accessibility.

---

## Key Tasks

1. **Plot ROC Curve and Compute AUC**: 
   After training the CNN, plot the ROC curve and compute the AUC (Area Under the Curve). This will be done both before and after training to assess model performance.
   
   - **True Positive Rate (TPR)**: The proportion of correctly predicted positive cases divided by the total positive cases.
   - **False Positive Rate (FPR)**: The proportion of incorrectly predicted positive cases divided by the total negative cases.
   
2. **Replicating Figure 3b**:  
   The first convolutional layer filters should learn to recognize specific DNA motifs. We will analyze these filters to confirm that the model is learning relevant biological motifs.

