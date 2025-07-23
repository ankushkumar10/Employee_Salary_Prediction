import streamlit as st
import pandas as pd
import joblib
import datetime
import seaborn as sns
import matplotlib.pyplot as plt

# --- 1. Load Model and Data ---

try:
    pipeline = joblib.load('salary_prediction_pipeline.pkl')
except FileNotFoundError:
    st.error("Model file not found. Please run train_model.py first.")
    st.stop()

# Load the raw dataset for the exploration page
try:
    df_raw = pd.read_csv('data/salary_dataset.csv')
except FileNotFoundError:
    df_raw = None

# --- 2. Page Configuration ---
st.set_page_config(
    page_title="Salary Prediction App",
    page_icon="💼",
    layout="wide"
)

# --- 3. Sidebar Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Salary Prediction", "Data Exploration", "About"])

# --- 4. Prediction Page ---
if page == "Salary Prediction":
    st.title("👨‍💻 Employee Salary Prediction")
    st.markdown("Enter the employee's details to predict their salary.")

    # Create columns for layout
    col1, col2 = st.columns(2)

    with col1:
        # Input fields for prediction
        age = st.number_input('Age', min_value=18, max_value=70, value=30, step=1)
        department = st.selectbox('Department', ['Sales', 'IT', 'HR'])
        experience = st.slider('Experience (years)', min_value=0, max_value=40, value=5, step=1)
        gender = st.selectbox('Gender', ['Male', 'Female', 'Other'])
        location = st.selectbox('Location', ['New York', 'Chicago', 'Los Angeles'])

    with col2:
        performance_score = st.slider('Performance Score', min_value=1.0, max_value=5.0, value=3.0, step=0.1)
        session = st.selectbox('Work Session', ['Morning', 'Evening', 'Night'])
        joining_date = st.date_input('Joining Date', value=datetime.date(2020, 1, 1))

    # Prediction button
    if st.button('Predict Salary', type="primary"):
        # Extract date components
        joining_year = joining_date.year
        joining_month = joining_date.month
        joining_day = joining_date.day

        # Create a DataFrame from inputs
        input_data = pd.DataFrame({
            'Age': [age],
            'Gender': [gender],
            'Department': [department],
            'Performance Score': [performance_score],
            'Experience': [experience],
            'Location': [location],
            'Session': [session],
            'Joining_Year': [joining_year],
            'Joining_Month': [joining_month],
            'Joining_Day': [joining_day]
        })
        
        # Make prediction
        prediction = pipeline.predict(input_data)[0]

        # Display the result
        st.success(f"Predicted Salary: ${prediction:,.2f}")


# --- 5. Data Exploration Page ---
elif page == "Data Exploration":
    st.title("📊 Data Exploration")
    st.markdown("Explore the underlying dataset used for training the model.")

    if df_raw is not None:
        # Show raw data
        if st.checkbox('Show Raw Data'):
            st.dataframe(df_raw)

        # Show summary statistics
        if st.checkbox('Show Summary Statistics'):
            st.write(df_raw.describe())
        
        st.subheader("Visualizations")
        
        # Distribution of Salary
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        sns.histplot(df_raw['Salary'], kde=True, ax=ax1, color='skyblue')
        ax1.set_title('Distribution of Salary')
        st.pyplot(fig1)

        # Salary by Department
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        sns.boxplot(x='Department', y='Salary', data=df_raw, ax=ax2)
        ax2.set_title('Salary by Department')
        st.pyplot(fig2)
        
        # Correlation Heatmap
        fig_corr, ax_corr = plt.subplots(figsize=(12, 9))
        correlation_matrix = df_raw.corr(numeric_only=True)
        sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', ax=ax_corr)
        ax_corr.set_title('Correlation Matrix of Numerical Features', fontsize=16)
        st.pyplot(fig_corr)

    else:
        st.warning("`salary_dataset.csv` not found. Cannot display data.")


# --- 6. About Page ---
elif page == "About":
    st.title("📄 About This Application")
    st.markdown("""
    This is a web application built to predict employee salaries based on several features.
    
    ### How it Works
    The prediction is made by a **Gradient Boosting Regressor** model, which was identified as the best-performing model during experimentation.
    
    The model was trained on the `salary_dataset.csv` file. The training process involves:
    - Handling missing values.
    - Engineering features from the 'Joining Date'.
    - Applying One-Hot Encoding to categorical variables like 'Department' and 'Location'.
    
    ### Technologies Used
    - **Python**: The core programming language.
    - **Streamlit**: For creating the interactive web application.
    - **Scikit-learn**: For building and training the machine learning model.
    - **Pandas**: For data manipulation.
    - **Seaborn & Matplotlib**: For data visualization on the 'Exploration' page.
    """)