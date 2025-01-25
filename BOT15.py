import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

st.title("Road Accident Severity Prediction")

# File uploader
data_file = st.file_uploader("Upload your Road Accidents CSV file", type=["csv"])

if data_file:
    try:
        # Load dataset with encoding handling
        data = pd.read_csv(data_file, encoding='latin-1')

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

    # --- Preprocessing and Feature Engineering ---

    # 1. Handle Missing Values (Adapt to your dataset's specifics)
    data.fillna(method='ffill', inplace=True) #Forward fill for simplicity.  Consider more robust methods.

    # 2. Identify Numerical and Categorical Features
    numeric_features = data.select_dtypes(include=np.number).columns
    categorical_features = data.select_dtypes(exclude=np.number).columns

    # 3. Create a Preprocessing Pipeline (Handles both types)
    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # 4. Define Features (X) and Target (y) (Adapt to your dataset)
    # Assuming 'accident_severity' is your target variable. Change if needed.
    if 'accident_severity' not in data.columns:
        st.error("Dataset must contain an 'accident_severity' column.")
        st.stop()
    
    X = data.drop('accident_severity', axis=1)
    y = data['accident_severity']

    # 5. Preprocess the data using the pipeline
    X_processed = preprocessor.fit_transform(X)

    # --- Train-Test Split and Model Training ---
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # --- Model Evaluation ---
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    st.subheader("Model Performance")
    st.write(f"Accuracy: {accuracy * 100:.2f}%")
    st.write("Classification Report:\n", report)

    # --- Prototype Prediction (Simple Example) ---
    st.subheader("Prototype Prediction")
    #Add input fields for relevant features here (replace with your actual features)
    #Example:
    example_features = {col: st.number_input(col, value=0.0) for col in X.columns}
    if st.button("Predict"):
        try:
            example_df = pd.DataFrame([example_features])
            example_processed = preprocessor.transform(example_df)
            prediction = model.predict(example_processed)
            st.write(f"Predicted Accident Severity: {prediction[0]}")
        except Exception as e:
            st.error(f"Error during prediction: {e}")

except Exception as e:
    st.error(f"An unexpected error occurred: {e}")