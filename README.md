# Resume Classifier with Streamlit Integration

This project demonstrates how to build a machine learning model that classifies resumes into different categories using text data. It includes a Streamlit web app for easy prediction of resume categories.

## Features

- Upload and process a CSV file containing resume data.
- Clean the resume text to remove unwanted characters and URLs.
- One-hot encode the resume categories.
- Vectorize the resume text using TfidfVectorizer.
- Train a K-Nearest Neighbors (KNN) classifier wrapped in a One-vs-Rest strategy.
- Evaluate the model's accuracy.
- Save and load the trained model and vectorizer for future predictions.
- Predict the category of a new resume via a Streamlit web app.


**Upload and Process Data**
    - The script will prompt you to upload a CSV file containing the resume data.
    - The CSV file should have at least two columns: 'Resume' and 'Category'.

**Data Cleaning**
    - The `cleanResume` function is used to clean the resume text by removing URLs, special characters, and extra spaces.

**One-Hot Encoding of Categories**
    - The `OneHotEncoder` from `sklearn` is used to one-hot encode the 'Category' column.

 **Text Vectorization**
    - The `TfidfVectorizer` is used to convert the cleaned resume text into a matrix of TF-IDF features.

**Model Training**
    - A K-Nearest Neighbors (KNN) classifier wrapped in a One-vs-Rest strategy is used to train the model on the vectorized text data.

To run streamlit app 

`streamlit run app.py`