import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path)

@st.cache_resource
def get_model(model_name):
    if model_name == "Random Forest":
        return RandomForestClassifier(n_estimators=10, max_depth=5, n_jobs=-1)  # Optimized for lightweight
    elif model_name == "CART":
        return DecisionTreeClassifier(max_depth=5)  # Optimized for lightweight
    else:
        raise ValueError("Unknown model name: {}".format(model_name))

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

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Preview Data", "Descriptive Statistics", "Classification and Comparison", "Prediction"])

# Initialize session state for data storage
if 'train_data' not in st.session_state:
    st.session_state.train_data = pd.DataFrame()
if 'test_data' not in st.session_state:
    st.session_state.test_data = pd.DataFrame()

# Paths to data files
TRAIN_DATA_FILE = "Data train balance.csv"
TEST_DATA_FILE = "Data test balance.csv"

if page == "Preview Data":
    st.header("Preview Data")

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

elif page == "Classification and Comparison":
    st.header("Classification and Comparison")

    if not st.session_state.train_data.empty and not st.session_state.test_data.empty:
        tab = st.radio("Select Option", ["Classification Models", "Comparison"], horizontal=True)

        if tab == "Classification Models":
            feature_columns = st.multiselect("Select Feature Columns (X)", st.session_state.train_data.columns)
            label_column = st.selectbox("Select Label Column (Y)", st.session_state.train_data.columns[::-1])  # Default to the last column

            if feature_columns and label_column:
                X_train = st.session_state.train_data[feature_columns]
                y_train = st.session_state.train_data[label_column]
                X_test = st.session_state.test_data[feature_columns]
                y_test = st.session_state.test_data[label_column]

                classifier_name = st.selectbox("Select Classifier", ["Random Forest", "CART"], index=0)
                model = get_model(classifier_name)
                model = train_model(model, X_train, y_train)
                y_pred = model.predict(X_test)

                accuracy = model.score(X_test, y_test)
                specificity = specificity_score(y_test, y_pred)
                sensitivity = sensitivity_score(y_test, y_pred)

                st.write("Accuracy: {:.3f}".format(accuracy))
                st.write("Sensitivity: {:.3f}".format(sensitivity))
                st.write("Specificity: {:.3f}".format(specificity))

                st.subheader("Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots()
                cax = ax.matshow(cm, cmap='Blues')
                plt.colorbar(cax)
                for (i, j), val in np.ndenumerate(cm):
                    ax.text(j, i, f'{val}', ha='center', va='center', color='red')
                plt.xlabel('Predicted')
                plt.ylabel('Actual')
                st.pyplot(fig)

                if classifier_name == "Random Forest":
                    st.subheader("Random Forest Tree Visualization")
                    fig, ax = plt.subplots(figsize=(12, 8))
                    plot_tree(model.estimators_[0], filled=True, ax=ax)  # Visualizing the first tree only
                    st.pyplot(fig)

                elif classifier_name == "CART":
                    st.subheader("Decision Tree Visualization")
                    fig, ax = plt.subplots(figsize=(12, 8))
                    _ = plot_tree(model, filled=True, ax=ax)
                    st.pyplot(fig)

        elif tab == "Comparison":
            feature_columns = st.multiselect("Select Feature Columns (X)", st.session_state.train_data.columns, key='comparison_features')
            label_column = st.selectbox("Select Label Column (Y)", st.session_state.train_data.columns[::-1], key='comparison_label')

            if feature_columns and label_column:
                X_train = st.session_state.train_data[feature_columns]
                y_train = st.session_state.train_data[label_column]
                X_test = st.session_state.test_data[feature_columns]
                y_test = st.session_state.test_data[label_column]

                classifiers = {
                    "Random Forest": get_model("Random Forest"),
                    "CART": get_model("CART")
                }

                metrics = []

                for name, model in classifiers.items():
                    model = train_model(model, X_train, y_train)
                    y_pred = model.predict(X_test)

                    accuracy = model.score(X_test, y_test)
                    specificity = specificity_score(y_test, y_pred)
                    sensitivity = sensitivity_score(y_test, y_pred)

                    metrics.append({
                        "Model": name,
                        "Accuracy": accuracy,
                        "Sensitivity": sensitivity,
                        "Specificity": specificity
                    })

                metrics_df = pd.DataFrame(metrics)
                st.write(metrics_df)

elif page == "Prediction":
    st.header("Prediction")

    if not st.session_state.train_data.empty and not st.session_state.test_data.empty:
        feature_columns = st.multiselect("Select Feature Columns (X)", st.session_state.train_data.columns, key='prediction_features')
        label_column = st.selectbox("Select Label Column (Y)", st.session_state.train_data.columns[::-1], key='prediction_label')
        classifier_name = st.selectbox("Select Classifier", ["Random Forest", "CART"], index=0, key='prediction_classifier')

        if feature_columns and label_column:
            X_train = st.session_state.train_data[feature_columns]
            y_train = st.session_state.train_data[label_column]

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
