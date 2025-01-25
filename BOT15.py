import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import seaborn as sns
import matplotlib.pyplot as plt

st.title("Road Accident Severity Prediction")

data_file = st.file_uploader("Upload your Road Accidents CSV file", type=["csv"])

if data_file is not None:
    try:
        # Load dataset with encoding handling
        df = pd.read_csv(data_file, encoding='utf-8', on_bad_lines='skip')

    except UnicodeDecodeError as e:
        st.error(f"Error decoding CSV file: {e}. Please check the file's encoding.")
        st.stop()
    except pd.errors.EmptyDataError:
        st.error("The CSV file is empty.")
        st.stop()
    except pd.errors.ParserError as e:
        st.error(f"Error parsing the CSV file: {e}. Please ensure it's a valid CSV.")
        st.stop()
    except FileNotFoundError:
        st.error("CSV file not found. Please upload a valid file.")
        st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred while loading the CSV: {e}")
        st.stop()

    #Data Cleaning and Preprocessing
    #Handle missing values (replace with appropriate strategy for your dataset)
    df.fillna(method='ffill', inplace=True)

    #Identify categorical and numerical columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    numerical_cols = df.select_dtypes(exclude=['object']).columns

    #Separate features (X) and target (y) - Adapt to your dataset
    if 'Accident_Severity' not in df.columns:
        st.error("Your dataset needs an 'Accident_Severity' column.")
        st.stop()
    X = df.drop('Accident_Severity', axis=1)
    y = df['Accident_Severity']

    #Preprocessing pipeline for categorical and numerical features
    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

    try:
        X_processed = preprocessor.fit_transform(X)
    except Exception as e:
        st.error(f"Error during preprocessing: {e}")
        st.stop()

    #Split the data
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

    #Train a Logistic Regression model
    model = LogisticRegression(max_iter=1000) #Increased max_iter to handle convergence issues
    try:
        model.fit(X_train, y_train)
    except Exception as e:
        st.error(f"Error during model training: {e}")
        st.stop()

    #Make predictions
    try:
        y_pred = model.predict(X_test)
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        st.stop()

    #Evaluate the model
    try:
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        st.subheader("Model Performance")
        st.write(f"Accuracy: {accuracy:.4f}")
        st.write("Classification Report:\n", report)
        st.write("Confusion Matrix:\n", cm)

        #Visualization of Confusion Matrix (optional)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        st.pyplot(plt)
    except Exception as e:
        st.error(f"Error during model evaluation: {e}")
        st.stop()

except Exception as e:
    st.error(f"An unexpected error occurred: {e}")