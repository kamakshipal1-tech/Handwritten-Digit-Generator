# ✍️ Handwritten Digit Generator using Deep Convolutional GAN (DCGAN)

## 📌 Project Overview

Generative Artificial Intelligence focuses on creating new data rather than simply analyzing existing data. One of the most powerful generative models is the **Generative Adversarial Network (GAN)**, where two neural networks compete against each other to generate realistic outputs.

This project implements a **Deep Convolutional Generative Adversarial Network (DCGAN)** using **TensorFlow and Keras** to generate realistic handwritten digits. The model is trained on the **MNIST handwritten digit dataset**, allowing it to learn the underlying distribution of handwritten numbers and create entirely new digit images from random noise.

A **Streamlit web application** has also been developed to allow users to generate new handwritten digits instantly with a single button click.

---

# 🎯 Business Problem

Deep learning models generally require a large amount of high-quality data for training. However, collecting and labeling image datasets is expensive, time-consuming, and sometimes impossible in many real-world applications.

Generative AI models can solve this challenge by learning the distribution of existing data and producing realistic synthetic images. These generated samples can be used for:

- Data augmentation
- Increasing dataset diversity
- Reducing manual data collection
- Training computer vision models
- Research in Generative AI
- Educational demonstrations of GANs

This project demonstrates how a DCGAN can generate realistic handwritten digits that closely resemble the original MNIST dataset.

---

# 🎯 Project Objectives

- Build a Deep Convolutional GAN (DCGAN)
- Learn handwritten digit patterns from the MNIST dataset
- Generate realistic synthetic handwritten digits
- Understand adversarial learning between Generator and Discriminator
- Deploy the trained Generator using Streamlit
- Create an interactive web application for digit generation

---

# 📊 Dataset Information

**Dataset:** MNIST Handwritten Digits

The MNIST dataset is one of the most widely used benchmark datasets for computer vision and deep learning.

### Dataset Statistics

- 60,000 Training Images
- 10,000 Testing Images
- Image Size: 28 × 28 pixels
- Grayscale Images
- 10 Classes (Digits 0–9)

The dataset is loaded directly using TensorFlow.

---

# ⚙ Data Preprocessing

Several preprocessing steps are performed before training the GAN.

## 1. Reshaping Images

Original Shape

```
(60000, 28, 28)
```

Converted Shape

```
(60000, 28, 28, 1)
```

The additional channel dimension is required by convolutional layers.

---

## 2. Image Normalization

Pixel values are scaled from

```
0 – 255
```

to

```
-1 to 1
```

using

```python
(train_images - 127.5) / 127.5
```

Normalization improves GAN training stability because the Generator uses a **tanh activation function**, whose output also lies between **-1 and 1**.

---

## 3. Dataset Pipeline

TensorFlow Dataset API is used for efficient training by

- Shuffling images
- Creating mini-batches
- Improving memory efficiency

Batch Size:

```
256
```

---

# 🧠 Understanding DCGAN

A **Deep Convolutional Generative Adversarial Network (DCGAN)** consists of two competing neural networks.

## Generator

The Generator receives a random noise vector of dimension **100**.

Its objective is to transform random noise into realistic handwritten digits capable of fooling the Discriminator.

---

## Discriminator

The Discriminator receives either

- A real handwritten digit
- A generated handwritten digit

Its objective is to determine whether the image is **Real** or **Fake**.

---

During training,

- The Generator continuously improves to fool the Discriminator.
- The Discriminator continuously improves to detect fake images.

This adversarial learning process gradually produces highly realistic handwritten digits.

---

# 🏗 Discriminator Architecture

The Discriminator performs binary classification.

### Architecture

- Conv2D
- LeakyReLU
- Dropout
- Conv2D
- LeakyReLU
- Dropout
- Flatten
- Dense Layer

Output

```
Single Prediction Score
```

Higher score indicates

```
Real Image
```

Lower score indicates

```
Fake Image
```

---

# 📉 Loss Function

The project uses **Binary Cross Entropy (BCE) Loss**.

Since the Discriminator performs binary classification

```
Real Image → 1

Fake Image → 0
```

Binary Cross Entropy measures how accurately the Discriminator distinguishes between real and generated images.

The Generator also uses Binary Cross Entropy Loss, but its objective is different. It tries to maximize the probability that generated images are classified as real by the Discriminator.

---

# ⚡ Optimizer

Both Generator and Discriminator use the **Adam Optimizer**.

Learning Rate

```
0.0001
```

Adam combines the advantages of

- Momentum
- Adaptive Learning Rate

Benefits include

- Faster convergence
- Stable GAN training
- Better optimization
- Efficient weight updates

---

# 🔄 Model Training

The training process consists of the following steps:

1. Generate a random noise vector.
2. Generator creates fake handwritten digits.
3. Real and fake images are passed to the Discriminator.
4. Discriminator predicts whether each image is real or fake.
5. Generator Loss and Discriminator Loss are calculated.
6. TensorFlow GradientTape computes gradients.
7. Adam Optimizer updates model parameters.
8. Sample images are generated after every epoch.
9. Model checkpoints are saved every 15 epochs.

The model is trained for **50 epochs**.

---

# 📈 Model Output

After training, the Generator successfully learns the distribution of handwritten digits and produces realistic synthetic images.

Every click on the Streamlit application generates a completely new set of handwritten digits because a different random noise vector is sampled each time.

---

# 🌐 Streamlit Web Application

A simple web application has been developed using **Streamlit**.

### Features

- Generate handwritten digits instantly
- Interactive user interface
- Displays 16 generated digits simultaneously
- Uses the trained Generator model
- No retraining required

---

# 🛠 Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Matplotlib
- Streamlit
- Google Colab
- Git
- GitHub

---

# 🚀 Future Improvements

- Conditional GAN (cGAN) for generating user-selected digits
- Train on Fashion-MNIST
- Generate higher-resolution handwritten images
- Implement Wasserstein GAN (WGAN)
- Deploy using Docker
- Deploy on cloud platforms
- Extend to colored image generation using CIFAR-10

---

# 📷 Application Preview

<img width="900" alt="Streamlit Application" src="YOUR_SCREENSHOT_HERE">

---

# ⭐ Conclusion

This project demonstrates the complete implementation of a **Deep Convolutional Generative Adversarial Network (DCGAN)** for handwritten digit generation. The Generator progressively learns to transform random noise into realistic handwritten digits, while the Discriminator simultaneously improves its ability to distinguish between real and generated images through adversarial learning.

In addition to understanding the fundamentals of Generative AI, this project provides hands-on experience with convolutional neural networks, TensorFlow model development, loss optimization, checkpointing, and deployment using Streamlit. The final application allows users to generate realistic handwritten digits interactively, making this an end-to-end deep learning project from data preprocessing and model training to deployment.
