# Credit Card Intrusion Detection System

## Overview
This project implements a Credit Card Intrusion Detection System using machine learning. The system is built with the Streamlit framework for creating interactive web applications and utilizes a RandomForestClassifier for fraud detection. The dataset used for training and testing is `creditcard.csv`.

## Files in the Project
1. **app.py**: The main application file containing the Streamlit web application code.
2. **creditcard.csv**: Dataset containing credit card transaction data.
3. **image.png**: Image file used in the web application.
4. **requirements.txt**: File listing the necessary dependencies for running the application.

## How to Run the Application
1. Install the required dependencies by running:
    ```
    pip install -r requirements.txt
    ```
2. Run the Streamlit application:
    ```
    streamlit run app.py
    ```

## Preview
### Interface Preview
<img width="946" alt="interface-preview" src="https://github.com/PrithaRajaguru/Credit-Card-Intrusion-Detection/assets/150375518/e7524e28-4424-4fb8-94c0-0a104e7592ab">

## Usage
1. Upon running the application, a web page titled "Credit Card Intrusion Detection System" will be displayed.
2. The page includes a section for generating random values and a section to input features for prediction.
3. Click the "Random" button to generate random values and observe the features displayed.
4. Enter the features in the input text box and click "Submit" to get a prediction on whether the transaction is legitimate or fraudulent.
5. The application will display the prediction and provide an audio output indicating whether the transaction is legitimate or fraudulent.

## Important Notes
- The dataset is initially loaded and balanced between legitimate and fraudulent transactions.
- The RandomForestClassifier is trained on the balanced dataset.
- Random values can be generated to observe the model's behavior.
- The application provides both visual and audio feedback for predictions.

## Acknowledgments
- This project is created by Anand and Pritha.
- Dataset source: [Credit Card Fraud Detection Dataset on Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

Feel free to customize this readme file based on your specific project details and requirements.
