# 💼 Employee Salary Prediction using Machine Learning & AI

## 📖 Overview

This project provides a web application for predicting employee salaries based on various features using **Machine Learning**. Built with **Streamlit**, it offers an interactive and user-friendly interface. At its core, it uses a trained **Gradient Boosting Regressor** model for accurate salary estimations, promoting fairness and data-driven compensation decisions.

---

## 🚀 Features

- **🎯 Salary Prediction:** Input employee details (Age, Department, Experience, Gender, Location, Performance Score, Work Session, Joining Date) to get an estimated salary.
- **📊 Data Exploration:** Visualize and explore the dataset (`salary_dataset.csv`) with summary statistics and interactive plots.
- **🖥️ Intuitive UI:** Simple and clean interface powered by Streamlit.

---

## 🧠 How It Works (Model & Algorithm)

The core model is a **Gradient Boosting Regressor**, selected for its superior performance in salary prediction.

### 🔄 Step-by-Step Process:

1. **Data Loading:** Load `salary_dataset.csv`.
2. **Data Preprocessing:**
   - Impute missing `Performance Score` with the median.
   - Transform `Joining Date` into `Joining_Year`, `Joining_Month`, and `Joining_Day`.
   - Drop irrelevant columns (`ID`, `Name`, `Status`).
3. **Feature Engineering:** Generate new date-based features.
4. **Data Splitting:** 80% training, 20% testing.
5. **Preprocessing Pipeline:**
   - `ColumnTransformer` applies `OneHotEncoder` to categorical features: `Gender`, `Department`, `Location`, `Session`.
   - Numerical features (`Age`, `Performance Score`, `Experience`, `Joining_Year`, `Joining_Month`, `Joining_Day`) are passed through directly.
6. **Model Training:** Use a `Pipeline` combining the preprocessor and `GradientBoostingRegressor`.
7. **Model Saving:** Save as `salary_prediction_pipeline.pkl`.

---

## 🛠️ Technologies Used

- **Python**: Core language.
- **Streamlit**: Web interface.
- **Scikit-learn**: Machine learning algorithms and preprocessing.
- **Pandas**: Data manipulation.
- **NumPy**: Numerical computing.
- **Matplotlib & Seaborn**: Visualization.
- **Joblib**: Saving and loading models.

---

## ⚙️ Setup and Installation

To run this project locally:

```bash
# 1. Clone the repository
git clone [YOUR_GITHUB_REPO_LINK_HERE]
cd [YOUR_REPO_NAME]

# 2. Create and activate a virtual environment (recommended)
python -m venv venv

# On Windows
.\venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
