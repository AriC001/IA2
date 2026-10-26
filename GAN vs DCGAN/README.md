# FIF4 Face Generation with GANs and DCGANs

## English

### Project Overview

This project explores the generation of synthetic FIF4 player faces using Generative Adversarial Networks (GANs). Two approaches were implemented and compared: a traditional GAN and a Deep Convolutional GAN (DCGAN).

The goal was to generate realistic and diverse face images while evaluating training performance, image quality, and training stability.

### Dataset

The project uses a dataset of FIF4 player face images, resized to **128 × 128 pixels** and normalized before training.

### Models

#### 1. Traditional GAN

A traditional GAN was implemented using fully connected neural network layers for the generator and discriminator.


#### Generator GAN

    GeneratorBlock(512, 256):   # 4x4 -> 8x8
        BatchNorm2d(512)
        ReLU()
        Upsample(scale_factor=2)
        Conv2d(512, 256, kernel=3x3, padding=1)
        BatchNorm2d(256)
        ReLU()
        Conv2d(256, 256, kernel=3x3, padding=1)
        Skip: Conv2d(512, 256, kernel=1x1)

    GeneratorBlock(256, 128)   # 8x8 -> 16x16
    GeneratorBlock(128, 64)    # 16x16 -> 32x32
    GeneratorBlock(64, 32)     # 32x32 -> 64x64
    GeneratorBlock(32, 32)     # 64x64 -> 128x128

    BatchNorm2d(32)
    ReLU()
    Conv2d(32, 3, kernel=3x3, padding=1)
    Tanh()


#### Discriminator GAN

    DiscriminatorBlock(64, 128):    # 128x128 -> 64x64
        ReLU()
        Conv2d(64, 128, kernel=3x3, padding=1) + SpectralNorm
        ReLU()
        Conv2d(128, 128, kernel=3x3, padding=1) + SpectralNorm
        Skip: Conv2d(64, 128, kernel=1x1) + SpectralNorm
        AvgPool2d(2)

    DiscriminatorBlock(128, 256)   # 64x64 -> 32x32
    DiscriminatorBlock(256, 512)   # 32x32 -> 16x16
    DiscriminatorBlock(512, 512)   # 16x16 -> 8x8

    ReLU()
    Flatten()
    Linear(512*8*8, 1)

** **
    latent_dim = 128
    Optimizer_G = Adam(lr=0.0001, betas=(0.0, 0.9))
    Optimizer_D = Adam(lr=0.00005, betas=(0.0, 0.9))

    Loss_D = Hinge Loss
    Loss_G = -mean(fake_scores)
    EMA_decay = 0.995

#### 2. DCGAN

A Deep Convolutional GAN (DCGAN) was implemented using convolutional layers to better capture spatial features in images.

<a name="generator"></a>
#### Generator

    ConvTranspose2d(100, 1024, kernel=4x4, stride=1, padding=0)  # 1x1 -> 4x4
    BatchNorm2d(1024)
    ReLU()

    ConvTranspose2d(1024, 512, kernel=4x4, stride=2, padding=1)  # 4x4 -> 8x8
    BatchNorm2d(512)
    ReLU()

    ConvTranspose2d(512, 256, kernel=4x4, stride=2, padding=1)  # 8x8 -> 16x16
    BatchNorm2d(256)
    ReLU()

    ConvTranspose2d(256, 128, kernel=4x4, stride=2, padding=1)  # 16x16 -> 32x32
    BatchNorm2d(128)
    ReLU()

    ConvTranspose2d(128, 64, kernel=4x4, stride=2, padding=1)  # 32x32 -> 64x64
    BatchNorm2d(64)
    ReLU()

    ConvTranspose2d(64, 3, kernel=4x4, stride=2, padding=1)  # 64x64 -> 128x128
    Tanh()

<a name="discriminator"></a>
#### Discriminator

    Conv2d(3, 64, kernel=4x4, stride=2, padding=1)  # 128x128 -> 64x64
    LeakyReLU(0.2)

    Conv2d(64, 128, kernel=4x4, stride=2, padding=1)  # 64x64 -> 32x32
    BatchNorm2d(128)
    LeakyReLU(0.2)

    Conv2d(128, 256, kernel=4x4, stride=2, padding=1)  # 32x32 -> 16x16
    BatchNorm2d(256)
    LeakyReLU(0.2)

    Conv2d(256, 512, kernel=4x4, stride=2, padding=1)  # 16x16 -> 8x8
    BatchNorm2d(512)
    LeakyReLU(0.2)

    Conv2d(512, 1024, kernel=4x4, stride=2, padding=1)  # 8x8 -> 4x4
    BatchNorm2d(1024)
    LeakyReLU(0.2)

    Conv2d(1024, 1, kernel=4x4, stride=1, padding=0)  # 4x4 -> 1x1


<a name="results"></a>
# Results
### Experiments and Results

Both architectures were trained and compared in terms of training time and generated image quality.

- **Traditional GAN:** 20 epochs took approximately 3 hours on Google Colab and 1 hour on the local GPU. **Total epochs = 60**
- **DCGAN:** 60 epochs took approximately 42 minutes on the local GPU. **Total epochs = 200**

The DCGAN produced more defined images and was considerably more computationally efficient in these experiments. However, training instability and mode collapse remained significant challenges.

**Results and generated images:**
## GAN
![epoch 1](<img/GAN epoch_001.png>)
![epoch 20](<img/GAN epoch_020.png>)
![epoch 40](<img/GAN epoch_040.png>)
![epoch 60](<img/GAN epoch_060.png>)

![results](<img/GAN 60.png>)
---
## DCGAN
![epoch 1](<img/DCGAN epoch_001.png>)
![epoch 50](<img/DCGAN epoch_050.png>)
![epoch 100](<img/DCGAN epoch_100.png>)
![epoch 200](<img/DCGAN epoch_200.png>)

![results](<img/DCGAN 200.png>)

### Conclusions

The experiments showed that convolutional architectures are more suitable for this image-generation task than the initial fully connected GAN implementation. Although DCGAN improved training efficiency and image definition, it did not completely solve training instability or mode collapse.

Future work could explore techniques such as Exponential Moving Average (EMA), data augmentation, and discriminator regularization to improve stability and image diversity.

---

## Español

### Descripción del proyecto

Este proyecto explora la generación de rostros sintéticos de jugadores de FIF4 mediante Redes Generativas Antagónicas (GAN). Se implementaron y compararon dos enfoques: una GAN tradicional y una Deep Convolutional GAN (DCGAN).

El objetivo fue generar rostros realistas y variados, evaluando el rendimiento del entrenamiento, la calidad de las imágenes y la estabilidad del modelo.

### Dataset

Se utilizó un conjunto de imágenes de rostros de jugadores de FIF4, redimensionadas a **128 × 128 píxeles** y normalizadas antes del entrenamiento.

### Modelos

#### 1. GAN tradicional

Se implementó una GAN tradicional utilizando capas neuronales densas (*fully connected*) en el generador y el discriminador.

**Arquitectura del generador:**

[Generador](#generator-gan)

**Arquitectura del discriminador:**

[Discriminador](#discriminator-gan)

#### 2. DCGAN

Se implementó una Deep Convolutional GAN (DCGAN), utilizando capas convolucionales para aprender mejor las características espaciales de las imágenes.

**Arquitectura del generador:**

[Generador](#generator)

**Arquitectura del discriminador:**

[Discriminador](#discriminator)

### Experimentos y resultados

Ambas arquitecturas se entrenaron y compararon en términos de tiempo de entrenamiento y calidad de las imágenes generadas.

- **GAN tradicional:** 20 épocas demoraron aproximadamente 3 horas en Google Colab y 1 hora en la GPU local. **Total épocas = 60** 
- **DCGAN:** 60 épocas demoraron aproximadamente 42 minutos en la GPU local. **Total épocas = 200** 

En estos experimentos, la DCGAN generó imágenes más definidas y fue considerablemente más eficiente en términos computacionales. Sin embargo, persistieron problemas de inestabilidad durante el entrenamiento y colapso de modos.

[Resultados e imágenes generadas](#results)

### Conclusiones

Los experimentos mostraron que las arquitecturas convolucionales son más adecuadas para esta tarea de generación de imágenes que la implementación inicial de GAN con capas densas. Aunque la DCGAN mejoró la eficiencia del entrenamiento y la definición de las imágenes, no logró resolver por completo la inestabilidad ni el colapso de modos.

Como trabajo futuro, se podrían explorar técnicas como Exponential Moving Average (EMA), aumentación de datos y regularización del discriminador para mejorar la estabilidad y la diversidad de las imágenes generadas.
