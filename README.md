# CDU - Clustering in Diversity and Uncertainty for Pavement Defect Detection

**Implementation of paper** - [CDU: A Diversity and Uncertainty-Based Active Learning Framework for Pavement Defect Detection]

[![arXiv](https://img.shields.io/badge/arXiv-2402.13616-b31b1b.svg)](https://arxiv.org/abs/2402.13616)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Spaces-blue.svg)](https://huggingface.co/spaces)
[![Google Colab](https://img.shields.io/badge/Open%20in%20Colab-yellowgreen)](https://colab.research.google.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Enabled-blue.svg)](https://opencv.org/)
[![BlogPost](https://img.shields.io/badge/BlogPost-Read-brightgreen.svg)](https://example.com/blogpost)

---

## Performance on Pavement Defect Detection Dataset

![Performance on Pavement Defect Dataset](performance_chart.png)

CDU achieves superior performance by selectively choosing the most informative samples for labeling. Leveraging the YOLOv9 backbone, CDU significantly improves defect detection in challenging, unbalanced datasets, ensuring high accuracy with fewer labeled samples.

---

## Performance
| Model           | Backbone     | Dataset             | mAP (%)  | Precision (%) | Recall (%) |
|-----------------|--------------|---------------------|----------|---------------|------------|
| CDU             | YOLOv9       | Pavement Defects    | 49.7     | 50.8          | 52.1       |
| CDU (Baseline)  | YOLOv8       | Pavement Defects    | 47.0     | 48.3          | 50.2       |
| Random          | YOLOv9       | Pavement Defects    | 43.2     | 44.5          | 47.8       |

---

## Description
This repository provides the **Clustering in Diversity and Uncertainty (CDU)** approach for optimizing active learning in object detection, specifically for detecting pavement defects like potholes, cracks, and patches. By using a YOLOv9 backbone, CDU selectively labels only the most informative images, significantly reducing data annotation costs while maintaining high accuracy.

### Key Features
- **Backbone Model**: Based on YOLOv9, which uses GELAN and Programmable Gradient Information (PGI) to optimize feature extraction for road defects.
- **Active Learning with CDU**: Combines uncertainty and diversity-based clustering to select samples that maximize model improvement with fewer labeled images.
- **Uncertainty and Confidence Calculation**: Uses custom uncertainty measures to prioritize the most informative samples in unbalanced datasets.

---

## Running on Google Colab
The entire CDU pipeline is available in Google Colab. **Only the Ultralytics YOLO library is required**, making setup quick and easy.

```python
# Install YOLOv9 from Ultralytics in Google Colab
!pip install ultralytics
