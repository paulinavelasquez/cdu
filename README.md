# CDU - Clustering in Diversity and Uncertainty for Pavement Defect Detection

**Implementation of paper** - [Active Learning Applied to YOLOv9 for the Detection and Classification of Pavement Defects]

[![arXiv](https://img.shields.io/badge/arXiv-2402.13616-b31b1b.svg)](https://arxiv.org/abs/2402.13616)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Spaces-blue.svg)](https://huggingface.co/spaces)
[![Google Colab](https://img.shields.io/badge/Open%20in%20Colab-yellowgreen)](https://colab.research.google.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Enabled-blue.svg)](https://opencv.org/)
[![BlogPost](https://img.shields.io/badge/BlogPost-Read-brightgreen.svg)](https://example.com/blogpost)

---

## Overview

This repository provides the **Clustering in Diversity and Uncertainty (CDU)** approach for optimizing active learning in object detection, focusing on pavement defects like potholes, cracks, and patches. By leveraging YOLOv9 as the backbone, CDU selectively labels only the most informative images, significantly reducing data annotation costs while ensuring high model pecision.

---

## Performance on Pavement Defect Detection Dataset

![Performance on Pavement Defect Dataset](performance_chart.png)

CDU demonstrates superior performance by selecting diverse and high-uncertainty samples, allowing it to maintain high detection accuracy with fewer labeled samples, even in unbalanced datasets.

---

## Performance Comparison of Methods

The table below compares the performance of different methods on the object detection task:

| Método   | Objetos | Precisión  | mAP@50     | Crack (mAP) | Patch (mAP) | Pothole (mAP) | AUC      |
|----------|---------|------------|------------|-------------|-------------|---------------|----------|
| Inicial  | 435     | 0.308090   | 0.204169   | 0.005444    | 0.354748    | 0.252317      | -        |
| Random   | 1873    | 0.431776   | 0.325667   | 0.021464    | 0.543822    | 0.411717      | 0.775421 |
| Sum      | 3486    | 0.465734   | 0.374198   | 0.025075    | 0.626333    | 0.471185      | 0.768690 |
| Avg      | 2159    | 0.454143   | **0.407099** | 0.032127 | **0.647408** | **0.541762** | 0.779141 |
| DUA      | 2598    | 0.446610   | 0.386136   | 0.035419    | 0.628757    | 0.494233      | 0.808191 |
| **CDU**  | 2441    | **0.497016** | 0.374519 | **0.044482** | 0.593338   | 0.485738      | **0.811746** |

---

## Landscape Overview

### Backbone Model: YOLOv9
CDU uses **YOLOv9** as its backbone model, a state-of-the-art single-stage object detector that is optimized for real-time applications. YOLOv9 includes enhancements such as **GELAN** and **Programmable Gradient Information (PGI)**, which allow for efficient feature retention even in deep networks. This helps the model handle complex road surfaces and varied defect types effectively.

### Confidence and Uncertainty Calculation

To determine which samples are the most informative for labeling, CDU calculates **confidence** and **uncertainty** for each detected object as follows:

- **Bounding Box Confidence** (`confidence_box`): Represents the likelihood of an object within a bounding box, calculated as:
  ![confidence_box](https://latex.codecogs.com/png.latex?confidence\_box%20%3D%20Pr(object)%20*%20IoU)

  
  
  where IoU (Intersection over Union) measures the overlap between the predicted bounding box and the ground truth.

- **Class Confidence** (`confidence_class`): Combines bounding box confidence with class probability, allowing YOLOv9 to determine the most likely class for each bounding box:
  confidence_class = max_k (confidence_box * Pr(class_k | object))

- **Uncertainty Score** (`uncertainty_object`): Defined as the complement of the class confidence, representing the uncertainty of the model’s prediction:
  uncertainty_object = 1 - confidence_class

These calculations allow CDU to prioritize high-uncertainty samples, optimizing the active learning process by selecting the most informative samples for labeling.

---

## Methodology

### 1. Diverse Uncertainty Aggregation (DUA)
The **DUA** score aggregates uncertainties across detected classes in each image, ensuring a balanced selection from different defect types. This prevents over-representation of one class in the labeled dataset.

### 2. Clustering in Diversity and Uncertainty (CDU)

CDU optimizes active learning through clustering, using **Gaussian Mixture Model (GMM)** to group images based on their class-specific uncertainty scores. The steps involved are:

- **Image Clustering**: Images are grouped based on uncertainty scores to ensure that selected samples represent diverse types of pavement defects.
- **Sainte-Laguë Method for Proportional Selection**: Ensures a balanced representation of clusters in the selected samples by allocating samples based on cluster size.
- **Modular Optimization**: A scoring function combines **uncertainty** and **diversity** in each sample batch, ensuring that the samples selected are both varied and informative. 

### 3. Sample Selection and Labeling
Using the scoring function, CDU prioritizes the samples that maximize learning gains in each active learning iteration, ensuring that the selected samples contribute effectively to model improvement.

---

## Running on Google Colab

The CDU framework is designed to run seamlessly on **Google Colab**. Only the **Ultralytics YOLO** library needs to be installed, making setup quick and easy.

```python
# Install YOLOv9 from Ultralytics in Google Colab
!pip install ultralytics

