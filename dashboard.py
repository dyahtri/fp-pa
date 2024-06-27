import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Fungsi untuk memuat data dengan pengaturan yang lebih efisien
@st.cache
def load_data(file_path):
    data = pd.read_csv(file_path, low_memory=True)
    return data

# Fungsi untuk memuat model
@st.cache
def get_model(model_name):
    if model_name == "Random Forest":
        return RandomForestClassifier(n_estimators=10)
    else:
        return DecisionTreeClassifier()

# Fungsi untuk melatih model
def train_model(model, X_train, y_train, model_path):
    model.fit(X_train, y_train)
    joblib.dump(model, model_path)
    return model

# Fungsi untuk memuat model yang sudah dilatih
def load_model(model_path):
    return joblib.load(model_path)

# Fungsi untuk menghitung skor khususitas (specificity) dari confusion matrix
def specificity_score(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp)
    return specificity

# Fungsi untuk menghitung skor sensitivitas (sensitivity) dari confusion matrix
def sensitivity_score(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn)
    return sensitivity

# Konfigurasi halaman Streamlit
st.set_page_config(page_title="Random Forest and CART Classification Dashboard", layout="wide")
st.title("Dashboard for Random Forest and CART Classification")

# Sidebar navigasi
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Preview Data", "Descriptive Statistics", "Classification and Comparison", "Prediction"])

# Inisialisasi state session untuk data latih dan uji
if 'train_data' not in st.session_state:
    st.session_state.train_data = pd.DataFrame()
if 'test_data' not in st.session_state:
    st.session_state.test_data = pd.DataFrame()

# File data latih dan uji
TRAIN_DATA_FILE = "Data train balance.csv"
TEST_DATA_FILE = "Data test balance.csv"

# Halaman untuk memuat data latih dan uji
if page == "Preview Data":
    st.header("Preview Data")

    try:
        st.session_state.train_data = load_data(TRAIN_DATA_FILE).sample(frac=0.1, random_state=1) # Memuat hanya sebagian data latih
        st.write("Training Data Preview")
        st.write(st.session_state.train_data.head())
    except FileNotFoundError:
        st.write(f"Training data file {TRAIN_DATA_FILE} not found.")

    try:
        st.session_state.test_data = load_data(TEST_DATA_FILE).sample(frac=0.1, random_state=1) # Memuat hanya sebagian data uji
        st.write("Testing Data Preview")
        st.write(st.session_state.test_data.head())
    except FileNotFoundError:
        st.write(f"Testing data file {TEST_DATA_FILE} not found.")

# Halaman untuk statistik deskriptif
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

# Halaman untuk klasifikasi dan perbandingan model
elif page == "Classification and Comparison":
    st.header("Classification and Comparison")

    if not st.session_state.train_data.empty and not st.session_state.test_data.empty:
        tab = st.radio("Select Option", ["Classification Models", "Comparison"], key='classification_comparison')

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
                model_path = f"{classifier_name.lower().replace(' ', '_')}_model.joblib"
                
                model = get_model(classifier_name)
                model = train_model(model, X_train, y_train, model_path)
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

                for name, model in classifiers.items():
                    model = train_model(model, X_train, y_train, f"{name.lower().replace(' ', '_')}_model.joblib")
                    y_pred = model.predict(X_test)

                    accuracy = model.score(X_test, y_test)
                    specificity = specificity_score(y_test, y_pred)
                    sensitivity = sensitivity_score(y_test, y_pred)

                    metrics.append({
                        "Model": name,
                        "Accuracy": accuracy,
                        "Sensitivity": sensitivity,
                        "Specificity": specificity,
                    })

                metrics_df = pd.DataFrame(metrics)
                st.write(metrics_df)

# Halaman untuk prediksi
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

            model_path = f"{classifier_name.lower().replace(' ', '_')}_model.joblib"
            model = load_model(model_path)

            st.subheader("Input Values for Prediction")
            input_data = {}
            for feature in feature_columns:
                input_value = st.number_input(f"Input value for {feature}", value=0.0)
                input_data[feature] = [input_value]

            input_df = pd.DataFrame(input_data)
            prediction = model.predict(input_df)[0]

            result = "Sah" if prediction == 0 else "Penipuan"

            st.write(f"Prediction: {result} (0: Sah, 1: Penipuan)")
