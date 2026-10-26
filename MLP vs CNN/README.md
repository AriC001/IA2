# README.md (Bilingüe / Bilingual)


# Telugu Handwritten Digits Classification: MLP vs. CNN
# Clasificación de Dígitos Manuscritos en Télugu: MLP vs. CNN

[English](#english) | [Español](#español)

---

<a name="english"></a>
## English

### Overview
This repository contains two Python notebooks implementing deep learning models to recognize handwritten digits from the **Telugu** script. The project compares a traditional **Multilayer Perceptron (MLP)** baseline against a spatial feature-extracting **Convolutional Neural Network (CNN)**.

### Dataset
* **Domain:** Handwritten Telugu numerals (0-9).
* **Format:** image folders (Kaggle Telugu Digits dataset).
* **Preprocessing:** Grayscale conversion, normalization (0 to 1)

### Repository Structure

```


│── HMR_MLP.ipynb    # Multilayer Perceptron 
└── HRM_CNN.ipynb    # Convolutional Neural Network 
└── README.md

```

## Model Architectures


### **MLP (Multilayer Perceptron)**

#### Model 1
    Input(shape=(28, 28, 1)),
    Flatten(),
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dense (10, activation='softmax')

#### Model 2

    Input(shape=(28, 28, 1)),
    Flatten(),
    Dense(512, activation='tanh'),
    Dense(256, activation='tanh'),
    Dense(128, activation='tanh'),
    Dense(10, activation='softmax')


#### Model 3

    Input(shape=(28, 28, 1)),
    Flatten(),
    Dense(128, activation='sigmoid'),
    Dense(64, activation='sigmoid'),
    Dense(10, activation='softmax')

### **CNN (Convolutional Neural Network):**

#### Model 1
    Input(shape=(28, 28, 3)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')

#### Model 2
    Input(shape=(28, 28, 1)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')

#### Model 3
    Input(shape=(28, 28, 1)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')

---
<a name="results"></a>
# Results
## MLP
### Model 1

![alt text](<img/Model MLP 1 Accuracy.png>)

![alt text](<img/Model MLP 1 Loss.png>)

Precision: 1.0000
Recall:    1.0000
F1 Score:  1.0000
| Number | precision | recall | f1-score | support|
| :--- | :---: | ---: | :--- | :---: |
|0|     1.0000|    1.0000|    1.0000|      2003|
|1|     1.0000|    1.0000|    1.0000|      2009|
|2|     1.0000|    1.0000|    1.0000|      1995|
|3|     1.0000|    1.0000|    1.0000|      1980|
|4|     1.0000|    1.0000|    1.0000|      2026|
|5|     1.0000|    1.0000|    1.0000|      1995|
|6|     1.0000|    1.0000|    1.0000|      2020|
|7|     1.0000|    1.0000|    1.0000|      1947|
|8|     1.0000|    1.0000|    1.0000|      2000|
|9|     1.0000|    1.0000|    1.0000|      2025|

![alt text](<img/Matriz Confusion MLP 1.png>)

### Model 2
![alt text](<img/Model MLP 2 Accuracy.png>)

![alt text](<img/Model MLP 2 Loss.png>)

Precision: 1.0000
Recall:    1.0000
F1 Score:  1.0000
| Number | precision | recall | f1-score | support|
| :--- | :---: | ---: | :--- | :---: |
|0|     1.0000|    1.0000|    1.0000|      2003|
|1|     1.0000|    1.0000|    1.0000|      2009|
|2|     1.0000|    1.0000|    1.0000|      1995|
|3|     1.0000|    1.0000|    1.0000|      1980|
|4|     1.0000|    1.0000|    1.0000|      2026|
|5|     1.0000|    1.0000|    1.0000|      1995|
|6|     1.0000|    1.0000|    1.0000|      2020|
|7|     1.0000|    1.0000|    1.0000|      1947|
|8|     1.0000|    1.0000|    1.0000|      2000|
|9|     1.0000|    1.0000|    1.0000|      2025|

![alt text](<img/Matriz Confusion MLP 2.png>)

### Model 3

![alt text](<img/Model MLP 3 Accuracy.png>)
![alt text](<img/Model MLP 3 Loss.png>)


Precision: 1.0000
Recall:    1.0000
F1 Score:  0.9999

| Number | precision | recall | f1-score | support|
| :--- | :---: | ---: | :--- | :---: |
|0|     1.0000|    0.9995|    0.9998|      2003|
|1|     0.9995|    1.0000|    0.9998|      2009|
|2|     1.0000|    1.0000|    1.0000|      1995|
|3|     1.0000|    1.0000|    1.0000|      1980|
|4|     1.0000|    1.0000|    1.0000|      2026|
|5|     1.0000|    1.0000|    1.0000|      1995|
|6|     1.0000|    1.0000|    1.0000|      2020|
|7|     1.0000|    1.0000|    1.0000|      1947|
|8|     1.0000|    1.0000|    1.0000|      2000|
|9|     1.0000|    1.0000|    1.0000|      2025|

![alt text](<img/Matriz Confusion MLP 3.png>)


## CNN

### Model 1
![alt text](<img/Model 1 Accuracy.png>)
![alt text](<img/Model 1 Loss.png>)
Precision: 1.0000
Recall:    1.0000
F1 Score:  1.0000

| Number | precision | recall | f1-score | support|
| :--- | :---: | ---: | :--- | :---: |
|0|     1.0000|    1.0000|    1.0000|      2003|
|1|     1.0000|    1.0000|    1.0000|      2009|
|2|     1.0000|    1.0000|    1.0000|      1995|
|3|     1.0000|    1.0000|    1.0000|      1980|
|4|     1.0000|    1.0000|    1.0000|      2026|
|5|     1.0000|    1.0000|    1.0000|      1995|
|6|     1.0000|    1.0000|    1.0000|      2020|
|7|     1.0000|    1.0000|    1.0000|      1947|
|8|     1.0000|    1.0000|    1.0000|      2000|
|9|     1.0000|    1.0000|    1.0000|      2025|


![alt text](<img/Matriz Confusion 1.png>)


### Model 2

![alt text](<img/Model 2 Accuracy.png>)

![alt text](<img/Model 2 Loss.png>)

Precision: 1.0000
Recall:    1.0000
F1 Score:  1.0000
| Number | precision | recall | f1-score | support|
| :--- | :---: | ---: | :--- | :---: |
|0|     0.9995|    1.0000|    0.9998|      2003|
|1|     1.0000|    0.9995|    0.9998|      2009|
|2|     1.0000|    1.0000|    1.0000|      1995|
|3|     1.0000|    1.0000|    1.0000|      1980|
|4|     1.0000|    1.0000|    1.0000|      2026|
|5|     1.0000|    1.0000|    1.0000|      1995|
|6|     1.0000|    1.0000|    1.0000|      2020|
|7|     1.0000|    1.0000|    1.0000|      1947|
|8|     1.0000|    1.0000|    1.0000|      2000|
|9|     1.0000|    1.0000|    1.0000|      2025|

![alt text](<img/Matriz Confusion 3.png>)
### Model 3

![alt text](<img/Model 3 Accuracy.png>)

![alt text](<img/Model 3 Loss.png>)

Precision: 1.0000
Recall:    1.0000
F1 Score:  1.0000
| Number | precision | recall | f1-score | support|
| :--- | :---: | ---: | :--- | :---: |
| 0  |   0.9995  | 1.0000  |  0.9998  |    2003|
| 1   |  1.0000   | 0.9995  |  0.9998    |  2009|
|2|     1.0000|    1.0000|    1.0000|      1995|
|3|     1.0000|    1.0000|    1.0000|      1980|
|4|     1.0000|    1.0000|    1.0000|      2026|
|5|     1.0000|    1.0000|    1.0000|      1995|
|6|     1.0000|    1.0000|    1.0000|      2020|
|7|     1.0000|    1.0000|    1.0000|      1947|
|8|     1.0000|    1.0000|    1.0000|      2000|
|9|     1.0000|    1.0000|    1.0000|      2025|


![Matriz text](<Matriz Confusion 3.png>)

---
<a name="español"></a>
## Español

### Descripción General

Este repositorio contiene dos notebooks de Python que implementan modelos de aprendizaje profundo para reconocer dígitos manuscritos en el alfabeto **Télugu**. El proyecto compara un modelo base tradicional de **Perceptrón Multicapa (MLP)** frente a una **Red Neuronal Convolucional (CNN)** especializada en la extracción de características espaciales.

### Conjunto de Datos (Dataset)

* **Dominio:** Numerales manuscritos en télugu (0-9).
* **Formato:** Carpetas de imágenes (dataset de Télugu de Kaggle).
* **Preprocesamiento:** Conversión a escala de grises, normalización (0 a 1),


## Arquitecturas de los Modelos

### **MLP (Perceptrón Multicapa)**

#### Modelo 1
    Input(shape=(28, 28, 1)),
    Flatten(),
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dense (10, activation='softmax')

#### Modelo 2

    Input(shape=(28, 28, 1)),
    Flatten(),
    Dense(512, activation='tanh'),
    Dense(256, activation='tanh'),
    Dense(128, activation='tanh'),
    Dense(10, activation='softmax')


#### Modelo 3

    Input(shape=(28, 28, 1)),
    Flatten(),
    Dense(128, activation='sigmoid'),
    Dense(64, activation='sigmoid'),
    Dense(10, activation='softmax')

### **CNN (Red Neuronal Convolucional):**
#### Modelo 1
    Input(shape=(28, 28, 3)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')

#### Modelo 2
    Input(shape=(28, 28, 1)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')

#### Modelo 3
    Input(shape=(28, 28, 1)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')

[Resultados](#results)