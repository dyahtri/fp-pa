import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, roc_curve, auc
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

@st.cache_data
def load_data(file_path):
    # Read data with only necessary columns and appropriate data types
    return pd.read_csv(file_path, dtype={'column1': 'float32', 'column2': 'int32', 'column3': 'category'})  # Example columns

@st.cache_resource
def get_model(model_name):
    if model_name == "Random Forest":
        return RandomForestClassifier(n_estimators=50, max_depth=3, n_jobs=-1)  # Reduced estimators and depth
    else:
        return DecisionTreeClassifier(max_depth=3)  # Reduced depth

def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model

def specificity_score(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp)
    return specificity

def sensitivity_score(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn)
    return sensitivity

st.set_page_config(page_title="Random Forest and CART Classification Dashboard", layout="wide")
st.title("Dashboard for Random Forest and CART Classification")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Preview Data", "Descriptive Statistics", "Classification and Comparison", "Prediction"])

if 'train_data' not in st.session_state:
    st.session_state.train_data = pd.DataFrame()
if 'test_data' not in st.session_state:
    st.session_state.test_data = pd.DataFrame()

TRAIN_DATA_FILE = "Data train balance.csv"
TEST_DATA_FILE = "Data test balance.csv"

if page == "Preview Data":
    st.header("Preview Data")

    try:
        st.session_state.train_data = load_data(TRAIN_DATA_FILE)
        st.write("Training Data Preview")
        st.write(st.session_state.train_data.head())
    except FileNotFoundError:
        st.write(f"Training data file {TRAIN_DATA_FILE} not found.")

    try:
        st.session_state.test_data = load_data(TEST_DATA_FILE)
        st.write("Testing Data Preview")
        st.write(st.session_state.test_data.head())
    except FileNotFoundError:
        st.write(f"Testing data file {TEST_DATA_FILE} not found.")

elif page == "Descriptive Statistics":
    st.header("Descriptive Statistics")

    if not st.session_state.train_data.empty:
        selected_columns_train = st.multiselect("Select Columns for Training Data (Descriptive Statistics)", st.session_state.train_data.columns)
        if selected_columns_train:
            st.write("Descriptive Statistics of Training Data")
            st.write(st.session_state.train_data[selected_columns_train].describe())

    if not st.session_state.test_data.empty:
        selected_columns_test = st.multiselect("Select Columns for Testing Data (Descriptive Statistics)", st.session_state.test_data.columns)
        if selected_columns_test:
            st.write("Descriptive Statistics of Testing Data")
            st.write(st.session_state.test_data[selected_columns_test].describe())

elif page == "Classification and Comparison":
    st.header("Classification and Comparison")

    if not st.session_state.train_data.empty and not st.session_state.test_data.empty:
        tab = st.radio("Select Option", ["Classification Models", "Comparison"], horizontal=True)

        if tab == "Classification Models":
            feature_columns = st.multiselect("Select Feature Columns (X)", st.session_state.train_data.columns)
            label_column = st.selectbox("Select Label Column (Y)", st.session_state.train_data.columns)

            if feature_columns and label_column:
                X_train = st.session_state.train_data[feature_columns]
                y_train = st.session_state.train_data[label_column]
                X_test = st.session_state.test_data[feature_columns]
                y_test = st.session_state.test_data[label_column]

                if y_train.dtype == 'O':
                    y_train = y_train.astype('category').cat.codes
                if y_test.dtype == 'O':
                    y_test = y_test.astype('category').cat.codes

                classifier_name = st.selectbox("Select Classifier", ["Random Forest", "CART"], index=0)
                model = get_model(classifier_name)
                model = train_model(model, X_train, y_train)
                y_pred = model.predict(X_test)

                st.subheader("Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                plt.xlabel('Predicted')
                plt.ylabel('Actual')
                st.pyplot(fig)

                if classifier_name == "Random Forest":
                    st.subheader("Random Forest Tree Visualization")
                    fig, ax = plt.subplots(figsize=(12, 8))
                    plot_tree(model.estimators_[0], filled=True, ax=ax)
                    st.pyplot(fig)

                elif classifier_name == "CART":
                    st.subheader("Decision Tree Visualization")
                    fig, ax = plt.subplots(figsize=(12, 8))
                    _ = plot_tree(model, filled=True, ax=ax)
                    st.pyplot(fig)

        elif tab == "Comparison":
            feature_columns = st.multiselect("Select Feature Columns (X)", st.session_state.train_data.columns, key='comparison_features')
            label_column = st.selectbox("Select Label Column (Y)", st.session_state.train_data.columns, key='comparison_label')

            if feature_columns and label_column:
                X_train = st.session_state.train_data[feature_columns]
                y_train = st.session_state.train_data[label_column]
                X_test = st.session_state.test_data[feature_columns]
                y_test = st.session_state.test_data[label_column]

                if y_train.dtype == 'O':
                    y_train = y_train.astype('category').cat.codes
                if y_test.dtype == 'O':
                    y_test = y_test.astype('category').cat.codes

                classifiers = {
                    "Random Forest": get_model("Random Forest"),
                    "CART": get_model("CART")
                }

                metrics = []
                roc_curves = {}

                for name, model in classifiers.items():
                    model = train_model(model, X_train, y_train)
                    y_pred = model.predict(X_test)

                    accuracy = model.score(X_test, y_test)
                    specificity = specificity_score(y_test, y_pred)
                    sensitivity = sensitivity_score(y_test, y_pred)
                    fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
                    roc_auc = auc(fpr, tpr)

                    metrics.append({
                        "Model": name,
                        "Accuracy": accuracy,
                        "Sensitivity": sensitivity,
                        "Specificity": specificity,
                        "AUC": roc_auc
                    })

                    roc_curves[name] = (fpr, tpr)

                metrics_df = pd.DataFrame(metrics)
                st.write(metrics_df)

                st.subheader("ROC Curves Comparison")
                fig = go.Figure()
                for name, (fpr, tpr) in roc_curves.items():
                    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'{name} ROC Curve'))
                fig.add_shape(type='line', x0=0, y0=0, x1=1, y1=1, line=dict(dash='dash', color='yellow'))
                fig.update_layout(xaxis_title='False Positive Rate', yaxis_title='True Positive Rate', title='ROC Curves Comparison')
                st.plotly_chart(fig)

elif page == "Prediction":
    st.header("Prediction")

    if not st.session_state.train_data.empty and not st.session_state.test_data.empty:
        feature_columns = st.multiselect("Select Feature Columns (X)", st.session_state.train_data.columns, key='prediction_features')
        label_column = st.selectbox("Select Label Column (Y)", st.session_state.train_data.columns, key='prediction_label')
        classifier_name = st.selectbox("Select Classifier", ["Random Forest", "CART"], index=0, key='prediction_classifier')

        if feature_columns and label_column:
            X_train = st.session_state.train_data[feature_columns]
            y_train = st.session_state.train_data[label_column]

            if y_train.dtype == 'O':
                y_train = y_train.astype('category').cat.codes

            model = get_model(classifier_name)
            model = train_model(model, X_train, y_train)

            st.subheader("Input Values for Prediction")
            input_data = {}
            for feature in feature_columns:
                input_value = st.number_input(f"Input value for {feature}", value=0.0)
                input_data[feature] = [input_value]

            input_df = pd.DataFrame(input_data)
            prediction = model.predict(input_df)[0]
            prediction_proba = model.predict_proba(input_df)[0]

            result = "Sah" if prediction == 0 else "Penipuan"

            st.write(f"Prediction: {result} (0: Sah, 1: Penipuan)")
            st.write(f"Prediction Probability: {prediction_proba}")
