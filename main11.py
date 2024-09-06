# import nessessary libraries
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_google_genai import ChatGoogleGenerativeAI

# page setup
st.set_page_config(page_title="AI Employee", page_icon="🤖")

st.title("🤖 AI Employee")
st.markdown("---")

# google gemini setup
api_key = 'YOUR_GOOGLE_API_KEY'
llm = ChatGoogleGenerativeAI(model='gemini-1.5-pro', google_api_key=api_key)

# file uploader
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    if "data" not in st.session_state:
        st.session_state["data"] = data

    # cleaning and encoding of text
    if "preprocessed" not in st.session_state:
        if st.button("Preprocess Data"):
            with st.spinner("Cleaning..."):
                for column in st.session_state["data"].select_dtypes(include=['object']).columns:
                    le = LabelEncoder()
                    st.session_state["data"][column] = le.fit_transform(st.session_state["data"][column].astype(str))

                st.session_state["data"].fillna(st.session_state["data"].median(), inplace=True)
                st.session_state["data"] = st.session_state["data"].dropna()
                st.session_state["data"] = st.session_state["data"].drop_duplicates()
            st.session_state["preprocessed"] = True
            st.success("Data preprocessing completed.")
            st.write("Preprocessed Data Preview:")
            st.dataframe(st.session_state["data"].head())

    # select type of prediction
    if "preprocessed" in st.session_state:
        task = st.selectbox("Select Prediction Type:", options=["Regression", "Classification"], key="task_selection")

        if "model_trained" not in st.session_state:
            if st.button("Train Model"):
                st.success("Searching for Patterns...")
                agent1 = create_csv_agent(llm, uploaded_file.name, verbose=False, handle_parsing_errors=True)
                tnp=agent1.invoke("Find the underlying trends and patterns in the csv file and summarize it in detail in a few bullet points")
                st.write(tnp['output'])
                
                X = st.session_state["data"].iloc[:, :-1]
                y = st.session_state["data"].iloc[:, -1]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

                if task == "Classification":
                    st.success("Training Classification Models...")
                    
                    ##RFC
                    st.subheader("Random Forest Classification (RFC)")
                    model = RandomForestClassifier(random_state=42)
                    model.fit(X_train,y_train)
                    y_pred = model.predict(X_test)
                    st.write("Accuracy Score:", accuracy_score(y_test, y_pred))
                    st.write("Classification Report:")
                    st.text(classification_report(y_test, y_pred))
                    
                    ## SVC
                    st.subheader("Support Vector Classification (SVC)")
                    svc_model = SVC(random_state=42)
                    svc_model.fit(X_train, y_train)
                    svc_pred = svc_model.predict(X_test)
                    st.write("Accuracy Score:", accuracy_score(y_test, svc_pred))
                    st.write("Classification Report:")
                    st.text(classification_report(y_test, svc_pred))

                    ## KNN
                    st.subheader("K-Nearest Neighbors (KNN)")
                    knn_model = KNeighborsClassifier()
                    knn_model.fit(X_train, y_train)
                    knn_pred = knn_model.predict(X_test)
                    st.write("Accuracy Score:", accuracy_score(y_test, knn_pred))
                    st.write("Classification Report:")
                    st.text(classification_report(y_test, knn_pred))

                elif task == "Regression":
                    st.success("Training Regression Models...")
                    
                    ##RFR
                    st.subheader("Random Forest Regressor (RFR)")
                    model = RandomForestRegressor(random_state=42)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    st.write("Mean Squared Error:", mean_squared_error(y_test, y_pred))
                    st.write("R-squared:", r2_score(y_test, y_pred))
                    
                    ## SVM
                    st.subheader("Support Vector Machine (SVM)")
                    svm_model = SVR()
                    svm_model.fit(X_train, y_train)
                    svm_pred = svm_model.predict(X_test)
                    st.write("Mean Squared Error:", mean_squared_error(y_test, svm_pred))
                    st.write("R-squared:", r2_score(y_test, svm_pred))

                    ## DT
                    st.subheader("Decision Tree Regression")
                    dt_model = DecisionTreeRegressor(random_state=42)
                    dt_model.fit(X_train, y_train)
                    dt_pred = dt_model.predict(X_test)
                    st.write("Mean Squared Error:", mean_squared_error(y_test, dt_pred))
                    st.write("R-squared:", r2_score(y_test, dt_pred))

                st.session_state["model_trained"] = True

        # report generation using LLM
        if "model_trained" in st.session_state and "report_generated" not in st.session_state:
            if st.button("Generate Report"):
                st.success("Creating Report...")
                agent2 = create_csv_agent(llm, uploaded_file.name, verbose=False, handle_parsing_errors=True)
                report = agent2.invoke("Give a long summary of the csv file. Then give a big report on the csv file by explaining it in a few bullet points after doing some exploratory data analysis.")
                st.write("Report:")
                st.write(report['output'])
                st.session_state["report_generated"] = True

        # generate plots
        if "report_generated" in st.session_state:
            if st.button("Generate Plots"):
                st.success("Creating Plots...")

                st.write("Correlation Matrix:")
                corr_matrix = st.session_state["data"].corr()
                fig, ax = plt.subplots(figsize=(12, 10))  
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
                st.pyplot(fig)

                st.write("Barplot of the dataset:")
                fig, ax = plt.subplots(figsize=(12, 8))  
                sns.barplot(data=st.session_state["data"].iloc[:, 1:], ax=ax)
                st.pyplot(fig)

                st.write("Boxplot of the dataset:")
                fig, ax = plt.subplots(figsize=(12, 8))  
                sns.boxplot(data=st.session_state["data"].iloc[:, 1:], ax=ax)
                st.pyplot(fig)

# exit button
if st.button("Exit"):
    st.success("Thank you for using!")
    for key in list(st.session_state.keys()):
        del st.session_state[key]
