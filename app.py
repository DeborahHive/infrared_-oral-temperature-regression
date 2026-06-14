import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns


class Config:
    PAGE_TITLE = "Temp ML App"
    PAGE_ICON = "🌡️"
    LAYOUT = "wide"

class ModelLoader:
    def __init__(self, path="final_tuned_model.pkl"):
        self.model = joblib.load(path)

    def predict(self, df):
        return self.model.predict(df)

class InputBuilder:
    def get_schema(self):
        return [
            "Gender", "Age", "Ethnicity",
            "T_atm", "Humidity", "Distance", "T_offset1",
            "Max1R13_1", "Max1L13_1", "aveAllR13_1", "aveAllL13_1",
            "T_RC1", "T_RC_Dry1", "T_RC_Wet1", "T_RC_Max1",
            "T_LC1", "T_LC_Dry1", "T_LC_Wet1", "T_LC_Max1",
            "RCC1", "LCC1", "canthiMax1", "canthi4Max1",
            "T_FHCC1", "T_FHRC1", "T_FHLC1", "T_FHBC1", "T_FHTC1",
            "T_FH_Max1", "T_FHC_Max1", "T_Max1", "T_OR1", "T_OR_Max1"
        ]

    def build(self, user_input: dict):
        cols = self.get_schema()

        df = pd.DataFrame([user_input])

        for c in cols:
            if c not in df.columns:
                df[c] = 0

        df = df[cols]

        df["Gender"] = df["Gender"].astype(str)
        df["Age"] = df["Age"].astype(str)
        df["Ethnicity"] = df["Ethnicity"].astype(str)

        numeric_cols = [c for c in cols if c not in ["Gender", "Age", "Ethnicity"]]

        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        return df


# Load Model
class StreamlitApp:

    def __init__(self):
        self.model = ModelLoader()

    def home_page(self):
        st.title("🌡️ Temperature Prediction App")
        st.write("""
        ## Welcome

    This machine learning application predicts a person's **oral body temperature**
    using infrared thermography measurements collected from different facial regions.

    Traditional temperature measurement methods require direct physical contact.
    Infrared thermography provides a non-contact alternative that can be useful in:

    - Healthcare screening
    - Fever detection
    - Smart hospitals
    - Public health monitoring
    - Remote patient assessment

    The model was trained using physiological, environmental, and thermal imaging
    features from the UCI Infrared Thermography Temperature Dataset.

    ### Features Used
    The prediction is based on:

    - Demographic information (Age, Gender, Ethnicity)
    - Environmental measurements (Atmospheric Temperature, Humidity)
    - Facial thermal measurements
    - Canthi region temperatures
    - Forehead temperature measurements
    - Thermal camera-derived statistics

    ### Machine Learning Models Evaluated

    During experimentation, multiple regression algorithms were compared:

    - Linear Regression
    - Ridge Regression
    - Random Forest Regressor
    - XGBoost Regressor

    The final deployed model was selected based on predictive performance and
    generalization ability.
    """)

        st.success("Best Model: Tuned Ridge Regression (R² ≈ 0.7508)")


class SideBar:
    def __init__(self, model):
        self.model = model

    def about_page(self):
        st.title("📘 About")

        st.write("""
        ## Project Overview

    Body temperature is one of the most important indicators of human health.
    Accurate temperature monitoring helps identify fever, infection, and other
    medical conditions.

    This project investigates whether machine learning can accurately estimate
    oral body temperature using infrared thermography data collected from
    different regions of the human face.

    ## Dataset

    Source:
    UCI Machine Learning Repository - Infrared Thermography Temperature Dataset.

    The dataset contains:

    - Demographic attributes
    - Environmental conditions
    - Facial thermal measurements
    - Oral temperature recordings

    ## Project Objective

    Build a machine learning regression model capable of predicting oral body
    temperature from thermal imaging measurements without direct physical contact.

    ## Technologies Used

    - Python
    - Pandas
    - Scikit-Learn
    - XGBoost
    - Streamlit
    - Joblib
        """)


    def method(self):
        st.title("⚙️ Methodology")

        st.markdown("""
    ## Project Workflow

    The machine learning pipeline followed the standard CRISP-DM process.

    ### 1. Data Understanding

    The dataset was explored to understand:

    - Feature distributions
    - Missing values
    - Relationships between variables
    - Correlations with oral temperature

    ### 2. Data Preprocessing

    Several preprocessing steps were applied:

    - Missing value imputation
    - Categorical feature encoding
    - Numerical feature scaling
    - Train-test splitting

    ### 3. Feature Engineering

    Features were organized into:

    - Demographic variables
    - Environmental measurements
    - Facial thermal measurements

    ### 4. Model Training

    Multiple regression algorithms were evaluated:

    - Linear Regression
    - Ridge Regression
    - Random Forest
    - XGBoost

    ### 5. Hyperparameter Tuning

    The best-performing models were tuned using cross-validation
    to improve predictive performance.

    ### 6. Model Selection

    Models were compared using:

    - R² Score
    - Mean Absolute Error (MAE)
    - Root Mean Squared Error (RMSE)

    The tuned Ridge Regression model achieved the best balance
    between accuracy and generalization and was selected for deployment.

        **Final model:** Ridge Regression
        """)
    
    def visualize(self):

        st.title("📊 Dataset Exploration")

        st.markdown("""
        This section provides insights into the dataset used to train
        the machine learning model.

        Exploratory Data Analysis (EDA) helps us understand:
        - Data quality
        - Feature distributions
        - Relationships between variables
        - Patterns affecting oral temperature prediction

        **Note**: Gender: Male-1, Female-0.
        """)

        try:

            df = pd.read_csv("dataset_preview.csv")

            st.subheader("Dataset Overview")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Rows", df.shape[0])

            with col2:
                st.metric("Columns", df.shape[1])

            with col3:
                st.metric("Missing Values", int(df.isnull().sum().sum()))

            st.divider()

            st.subheader("Dataset Preview")
            st.dataframe(df.head())

            st.divider()

            st.subheader("Statistical Summary")
            st.dataframe(df.describe())

            st.divider()

            st.subheader("Missing Values")

            missing = df.isnull().sum()
            missing = missing[missing > 0]

            if len(missing) > 0:
                st.dataframe(missing)
            else:
                st.success("No missing values found.")

            st.divider()

            st.subheader("Gender Distribution")

            if "Gender" in df.columns:

                fig, ax = plt.subplots()

                df["Gender"].value_counts().plot(
                    kind="bar",
                    ax=ax
                )

                ax.set_ylabel("Count")

                st.pyplot(fig)

            st.divider()

            st.subheader("Age Group Distribution")

            if "Age" in df.columns:

                fig, ax = plt.subplots()

                df["Age"].value_counts().plot(
                    kind="bar",
                    ax=ax
                )

                ax.set_ylabel("Count")

                st.pyplot(fig)

            st.divider()

            st.subheader("Correlation Heatmap")

            numeric_df = df.select_dtypes(include="number")

            fig, ax = plt.subplots(figsize=(12, 8))

            sns.heatmap(
                numeric_df.corr(),
                cmap="coolwarm",
                ax=ax
            )

            st.pyplot(fig)

            st.divider()

            st.subheader("Temperature Feature Distribution")

            temp_cols = [
                col for col in df.columns
                if col.startswith("T_")
            ]

            if len(temp_cols) > 0:

                selected = st.selectbox(
                    "Select Temperature Feature",
                    temp_cols
                )

                fig, ax = plt.subplots()

                sns.histplot(
                    df[selected],
                    kde=True,
                    ax=ax
                )

                ax.set_title(selected)

                st.pyplot(fig)

        except Exception as e:
            st.error(f"Error loading dataset: {e}")

    def predict(self):

        st.markdown("""
    ### Temperature Prediction

    Enter patient information and thermal sensor measurements below.

    The model will estimate the individual's oral body temperature based on
    patterns learned from the training dataset.

    **Note:**
    This application is intended for educational and research purposes and
    should not be used as a replacement for professional medical diagnosis.
""")
        
        st.title("🔮 Predict Temperature")

        schema_defaults = {
            "Gender": "Male",
            "Age": "21-30",
            "Ethnicity": "White",
            "T_atm": 24.0,
            "Humidity": 28.0,
            "Distance": 0.8,
            "T_offset1": 0.7,
            "Max1R13_1": 35.0,
            "Max1L13_1": 35.0,
            "aveAllR13_1": 34.0,
            "aveAllL13_1": 34.0,
            "T_RC1": 35.0,
            "T_RC_Dry1": 35.0,
            "T_RC_Wet1": 35.0,
            "T_RC_Max1": 35.0,
            "T_LC1": 35.0,
            "T_LC_Dry1": 35.0,
            "T_LC_Wet1": 35.0,
            "T_LC_Max1": 35.0,
            "RCC1": 34.0,
            "LCC1": 34.0,
            "canthiMax1": 34.0,
            "canthi4Max1": 34.0,
            "T_FHCC1": 34.0,
            "T_FHRC1": 34.0,
            "T_FHLC1": 34.0,
            "T_FHBC1": 34.0,
            "T_FHTC1": 34.0,
            "T_FH_Max1": 34.0,
            "T_FHC_Max1": 34.0,
            "T_Max1": 35.0,
            "T_OR1": 35.0,
            "T_OR_Max1": 35.0
        }

        user_input = {}

        user_input["Gender"] = st.selectbox("Gender", ["Male", "Female"])
        user_input["Age"] = st.selectbox("Age", ["18-20", "21-30", "31-40", "41-50"])
        user_input["Ethnicity"] = st.selectbox(
            "Ethnicity",
            ["White", "Black or African-American", "Asian", "Other"]
        )

        for k, v in schema_defaults.items():
            if k not in user_input:
                user_input[k] = st.number_input(k, value=float(v))

        if st.button("Predict"):
            builder = InputBuilder()
            df = builder.build(user_input)
            pred = self.model.predict(df)[0]
            st.success(f"Predicted Oral Temperature: {pred:.2f} °C")


class App:

    def __init__(self):

        st.set_page_config(
            page_title=Config.PAGE_TITLE,
            page_icon=Config.PAGE_ICON,
            layout=Config.LAYOUT
        )

    def run(self):
        app = StreamlitApp()
        nav = SideBar(app.model)


        home = st.Page(app.home_page, title="Home", icon=":material/home:", default=True)
        about = st.Page(nav.about_page, title="About", icon=":material/info:")
        methodology = st.Page(nav.method, title="Methodology", icon=":material/flowchart:")
        visualization = st.Page(nav.visualize, title="Visualization", icon=":material/bar_chart:")
        predict = st.Page(nav.predict, title="Predict Toxicity", icon=":material/biotech:")

        pg = st.navigation([home, about, methodology, visualization, predict])
        pg.run()

if __name__ == "__main__":
    App().run()