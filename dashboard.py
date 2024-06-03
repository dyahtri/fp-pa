import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, roc_curve, auc
import plotly.figure_factory as ff
import plotly.graph_objects as go
import matplotlib.pyplot as plt

@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path)

@st.cache_resource
def train_model(model_name, X_train, y_train):
    if model_name == "Random Forest":
        model = RandomForestClassifier()
    else:
        model = DecisionTreeClassifier()
    
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

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Load Data", "Descriptive Statistics", "Classification Models", "Prediction", "Comparison"])

# Initialize session state for data storage
if 'train_data' not in st.session_state:
    st.session_state.train_data = pd.DataFrame()
if 'test_data' not in st.session_state:
    st.session_state.test_data = pd.DataFrame()

# Paths to data files
TRAIN_DATA_FILE = "Data train balance.csv"
TEST_DATA_FILE = "Data test balance.csv"

if page == "Load Data":
    st.header("Load Data")

    # Load training data
    try:
        st.session_state.train_data = load_data(TRAIN_DATA_FILE)
        st.write("Training Data Preview")
        st.write(st.session_state.train_data.head())
    except FileNotFoundError:
        st.write(f"Training data file {TRAIN_DATA_FILE} not found.")

    # Load testing data
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

elif page == "Classification Models":
    st.header("Classification Models")

    if not st.session_state.train_data.empty and not st.session_state.test_data.empty:
        feature_columns = st.multiselect("Select Feature Columns (X)", st.session_state.train_data.columns)
        label_column = st.selectbox("Select Label Column (Y)", st.session_state.train_data.columns)
        classifier_name = st.selectbox("Select Classifier", ["Random Forest", "CART"], index=0)

        if feature_columns and label_column:
            X_train = st.session_state.train_data[feature_columns]
            y_train = st.session_state.train_data[label_column]
            X_test = st.session_state.test_data[feature_columns]
            y_test = st.session_state.test_data[label_column]

            model = train_model(classifier_name, X_train, y_train)
            y_pred = model.predict(X_test)

            accuracy = model.score(X_test, y_test)
            specificity = specificity_score(y_test, y_pred)
            sensitivity = sensitivity_score(y_test, y_pred)
            fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
            roc_auc = auc(fpr, tpr)

            st.write("Accuracy: {:.3f}".format(accuracy))
            st.write("Sensitivity: {:.3f}".format(sensitivity))
            st.write("Specificity: {:.3f}".format(specificity))
            st.write("AUC: {:.3f}".format(roc_auc))

            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig = ff.create_annotated_heatmap(cm, x=["Predicted Negative", "Predicted Positive"], y=["Actual Negative", "Actual Positive"])
            st.plotly_chart(fig)

            st.subheader("ROC Curve")
            fig = go.Figure(data=go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC Curve'))
            fig.add_shape(type='line', x0=0, y0=0, x1=1, y1=1, line=dict(dash='dash', color='yellow'))
            fig.update_layout(xaxis_title='False Positive Rate', yaxis_title='True Positive Rate', title='ROC Curve')
            st.plotly_chart(fig)

            if classifier_name == "CART":
                st.subheader("Decision Tree Visualization")
                fig, ax = plt.subplots(figsize=(12, 8))
                plot_tree(model, filled=True, ax=ax)
                st.pyplot(fig)
                
            elif classifier_name == "Random Forest":
                st.subheader("Random Forest Trees Visualization")
                num_trees_to_plot = min(3, len(model.estimators_))
                for i in range(num_trees_to_plot):
                    fig, ax = plt.subplots(figsize=(12, 8))
                    plot_tree(model.estimators_[i], filled=True, ax=ax)
                    st.pyplot(fig)

elif page == "Prediction":
    st.header("Prediction")

    if not st.session_state.train_data.empty and not st.session_state.test_data.empty:
        feature_columns = st.multiselect("Select Feature Columns (X)", st.session_state.train_data.columns, key='prediction_features')
        label_column = st.selectbox("Select Label Column (Y)", st.session_state.train_data.columns, key='prediction_label')
        classifier_name = st.selectbox("Select Classifier", ["Random Forest", "CART"], index=0, key='prediction_classifier')

        if feature_columns and label_column:
            X_train = st.session_state.train_data[feature_columns]
            y_train = st.session_state.train_data[label_column]

            model = train_model(classifier_name, X_train, y_train)

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

elif page == "Comparison":
    st.header("Comparison of Random Forest and CART")

    if not st.session_state.train_data.empty and not st.session_state.test_data.empty:
        feature_columns = st.multiselect("Select Feature Columns (X)", st.session_state.train_data.columns, key='comparison_features')
        label_column = st.selectbox("Select Label Column (Y)", st.session_state.train_data.columns, key='comparison_label')

        if feature_columns and label_column:
            X_train = st.session_state.train_data[feature_columns]
            y_train = st.session_state.train_data[label_column]
            X_test = st.session_state.test_data[feature_columns]
            y_test = st.session_state.test_data[label_column]

            classifiers = {
                "Random Forest": RandomForestClassifier(),
                "CART": DecisionTreeClassifier()
            }

            metrics = []
            roc_curves = {}

            for name, model in classifiers.items():
                model = train_model(name, X_train, y_train)
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
