# AI-Driven Emotion Detection from Text using Machine Learning

## Project Overview

AI-Driven Emotion Detection from Text using Machine Learning is an NLP-based machine learning project that automatically identifies the emotion expressed in a given text.

The system aims to classify text into emotions such as **joy, sadness, anger, fear, love, and surprise**. It uses text preprocessing and **TF-IDF vectorization** to convert textual data into numerical features, followed by a **Logistic Regression** model for emotion classification. MLflow is used for experiment tracking and model versioning, while Flask is used to deploy the trained model through a REST API.

---

## Team Members

| S. No. | University ID | Name            |
| ------ | ------------- | --------------- |
| 1      | 240030231     | Sneha Bomma     |
| 2      | 2420030445    | Lalitha mahisri |
| 3      | 2420030787    | Devaansh Sai    |

### Supervisor

**Dr. Archana Kalidindi**

---

## Abstract

The rapid growth of digital communication through social media, online reviews, messaging platforms, and customer feedback systems has generated a vast amount of textual data containing valuable emotional information. Manually analysing these emotions is time-consuming and inefficient.

This project presents an automated emotion detection system that classifies textual input into emotions such as joy, sadness, anger, fear, love, and surprise using Machine Learning and Natural Language Processing (NLP) techniques. The system preprocesses text, converts it into numerical features using the TF-IDF vectorization technique, and trains a Logistic Regression model to predict emotions accurately.

To introduce fundamental MLOps concepts, the project integrates MLflow for experiment tracking and model versioning, while Flask is used to deploy the trained model as a REST API. A simple web interface enables users to enter text and receive real-time emotion predictions. The project demonstrates an end-to-end workflow for building, managing, and deploying a machine learning model while maintaining simplicity suitable for academic learning.

---

## Technologies Used

* Python
* Machine Learning
* Natural Language Processing (NLP)
* TF-IDF Vectorization
* Logistic Regression
* Flask
* MLflow

---

## Setup and Installation

### 1. Clone the Repository

```bash
git clone <YOUR-GITHUB-REPOSITORY-URL>
cd <YOUR-PROJECT-FOLDER>
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install Required Dependencies

If a `requirements.txt` file is available:

```bash
pip install -r requirements.txt
```

Otherwise, install the required Python libraries used by the project:

```bash
pip install pandas numpy scikit-learn flask mlflow joblib
```

---

## Execution

### Train the Model

Run the project's model training script:

```bash
python <training-file-name>.py
```

### Run the Application

Start the Flask application:

```bash
python <flask-file-name>.py
```

Open the local URL displayed in the terminal in a web browser.

### MLflow

MLflow can be started using:

```bash
mlflow ui
```

It can be used to view experiment tracking and model-related information.

---

## Project Workflow

```text
Input Text
     ↓
Text Preprocessing
     ↓
TF-IDF Vectorization
     ↓
Logistic Regression Model
     ↓
Emotion Prediction
     ↓
Flask REST API
     ↓
Web Interface
```

---

## Current Phase Status

**Phase: Development / Implementation**

**Status: In Progress**

### Completed / Planned Components

* [x] Project topic and objectives defined
* [x] Emotion classification approach identified
* [x] NLP approach selected
* [x] TF-IDF vectorization selected
* [x] Logistic Regression selected
* [x] MLflow selected for experiment tracking and model versioning
* [x] Flask selected for API deployment
* [ ] Model training and evaluation
* [ ] Flask API integration
* [ ] Web interface integration
* [ ] End-to-end testing
* [ ] Final deployment
* [ ] Final documentation

---

## Expected Output

The system accepts textual input from the user and predicts the emotion expressed in the text.

For example:

```text
Input:
"I am extremely happy with the results!"

Predicted Emotion:
Joy
```

---

## Future Scope

* Improve classification accuracy.
* Support additional emotions.
* Experiment with different machine learning models.
* Use larger and more diverse datasets.
* Improve the web interface.
* Deploy the application online.
* Extend MLflow-based experiment tracking and model management.

---

## Academic Information

**Course:** Adaptive Software Engineering (24CI3201)
**Academic Year:** 2026–2027
**Department:** Computer Science and Engineering
