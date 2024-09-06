# AI Employee - Data Preprocessing and Model Training with Streamlit



## Data Preprocessing and Model Training Application

This repository contains a Streamlit application for data preprocessing, model training, and report generation. The application allows users to preprocess their data, select prediction types (regression or classification), train various machine learning models, and visualize results.

## Features

- **Data Preprocessing**: Clean and prepare your data by handling missing values and encoding categorical variables.
- **Model Training**: Train multiple machine learning models for both classification and regression tasks.
- **Performance Evaluation**: Get accuracy scores and detailed classification or regression reports.
- **Report Generation**: Generate a comprehensive report summarizing the data and model performance.
- **Data Visualization**: Create plots to visualize correlations and distributions in the dataset.

## Requirements/ Dependencies

To run this application, you will need the following Python packages:

- `streamlit`
- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `seaborn`

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Bhudil/AI_Employee.git
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```
3. Set up your Google API key:
- Obtain an API key from the [Google AI Studio](https://makersuite.google.com/app/apikey)
- Replace `'YOUR_GOOGLE_API'` in the code with your actual API key


## Usage

1. Run the Streamlit app:
```bash
streamlit run main11.py
```

1. **Upload Your CSV File:**
Upon launching the app, you will be prompted to upload a CSV file. This file will be the input for data preprocessing and model training.
2. **Data Preprocessing:**
Click on Preprocess Data to clean the dataset, handle missing values, encode categorical data, and display the preprocessed data preview.
3. **Model Training:**
Select either Classification or Regression from the dropdown.
Train the machine learning models by clicking on Train Model. The app supports Random Forest, SVM, KNN, and Decision Tree models for both classification and regression tasks.
4. **Report Generation:**
After training the model, generate an Exploratory Data Analysis (EDA) Report by clicking on Generate Report. This will invoke Langchain’s agent toolkit to generate an extensive analysis of the data and trends.
5. **Visualizations:**
Click Generate Plots to create visualizations, including a correlation heatmap, bar plot, and box plot of the dataset.

![Screenshot 2024-09-06 180832](https://github.com/user-attachments/assets/7ea4184b-e9af-4f44-83ad-ce0a042f1163)


## License

Distributed under the MIT License. See `LICENSE` for more information.

## Acknowledgements

- [Streamlit](https://streamlit.io/)
- [LangChain](https://python.langchain.com/)
- [Google Generative AI](https://cloud.google.com/ai/generative-ai)


