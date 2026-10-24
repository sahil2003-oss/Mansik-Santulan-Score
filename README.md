# Student Mental Health Score Predictor

An end-to-end Machine Learning web application designed to analyze student mental health datasets and predict scores based on behavioral metrics.

---

## Overview

This project processes student mental health data sourced from Kaggle. It implements a robust machine learning pipeline featuring thorough exploratory data analysis (EDA) and model benchmarking to predict mental health scores accurately.

---

## Key Features

* **Data Engineering & EDA**: Conducted comprehensive data cleaning, outlier detection, and skewness/distribution analysis.
* **Leak-Free Pipelines**: Built reproducible preprocessing and training pipelines using `Column Transformer` for integrated feature scaling and encoding.
* **Model Benchmarking**: Compared linear baseline models against tree-based ensembles (Linear Regression vs. Random Forest Regressor) with hyperparameter tuning via `RandomizedSearchCV`.
* **Performance**: Achieved a best test R2 score of **0.8714** using a tuned Random Forest Regressor.
* **Live Deployment**: Hosted and deployed as a live web application on Render.

---

## Tech Stack

* **Language**: Python
* **Libraries**: scikit-learn, Pandas, NumPy
* **Deployment**: Render

---

## Getting Started

### Prerequisites

* Python 3.x installed on your system.

### Installation & Setup

1. **Clone the Repository**:
   ```bash
   git clone [https://github.com/sahil2003-oss/Mansik-Santulan-Score.git](https://github.com/sahil2003-oss/Mansik-Santulan-Score.git)
   cd Mansik-Santulan-Score

2. **Install the Dependencies**:
   ```bash
   pip install -r requirements.txt

3. **Run the Application**:
   ```bash
   python main.py
