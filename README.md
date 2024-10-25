# Brain Tumor Classification System

## Overview

This project is an automated brain tumor classification system using machine learning techniques. It leverages Histogram of Oriented Gradients (HOG) feature extraction and a Support Vector Machine (SVM) classifier to classify MRI images into four categories: glioma, meningioma, pituitary, and no tumor. The system aims to assist clinicians in accurately diagnosing and treating various types of brain tumors.

## Live Demo

You can access the live demo of the application at the following link:

[Live Demo](https://braintumorclassification-wogrbfcfmb3ucqmcyxwqsl.streamlit.app/)

## Features

- **Image Classification**: Classifies brain tumor images into different categories.
- **User-Friendly Interface**: Built using Streamlit for easy interaction.
- **Real-Time Predictions**: Upload images and get instant predictions.
- **Visualization**: Displays results and predictions visually.

## Installation

To run this project locally, follow the steps below:

1. Clone the repository:
    ```bash
    git clone <REPOSITORY_URL>
    cd <REPOSITORY_DIRECTORY>
    ```

2. Create a virtual environment (optional but recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To start the application, run the following command:

```bash
streamlit run app.py
