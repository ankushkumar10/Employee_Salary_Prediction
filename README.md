# Employee Salary Prediction using AI & ML

## 📌 Overview

This project provides a web application for predicting employee salaries based on various features using **Machine Learning**. The application is built with **Streamlit**, making it interactive and easy to use. It leverages a trained **Gradient Boosting Regressor** model to offer accurate salary estimations, promoting fairness and data-driven decision-making in compensation.

---

## 🚀 Features

- **Salary Prediction**: Input employee details (Age, Department, Experience, Gender, Location, Performance Score, Work Session, Joining Date) to get an estimated salary.
- **Data Exploration**: Visualize and understand the underlying dataset (`salary_dataset.csv`) through interactive plots and summary statistics.
- **User-Friendly Interface**: An intuitive web interface built with Streamlit for seamless interaction.

---

## ⚙️ How It Works (Model & Algorithm)

The core of this application is a **Gradient Boosting Regressor** model, selected for its performance in predicting salaries. The entire ML pipeline, from data preprocessing to model training, is encapsulated for efficient deployment.

### 🔄 Step-by-Step Process:

1. **Data Loading**: Load `salary_dataset.csv`.
2. **Data Preprocessing**:
   - Impute missing 'Performance Score' values using the median.
   - Convert 'Joining Date' into `Joining_Year`, `Joining_Month`, `Joining_Day`.
   - Drop irrelevant columns (`ID`, `Name`, `Status`).
3. **Feature Engineering**: Create new date-based features.
4. **Data Splitting**: 80% training and 20% testing.
5. **Preprocessing Pipeline**:
   - `OneHotEncoder` for categorical features: `Gender`, `Department`, `Location`, `Session`.
   - Pass numerical features directly: `Age`, `Performance Score`, `Experience`, `Joining_Year`, `Joining_Month`, `Joining_Day`.
6. **Model Training**: Combine preprocessor with `GradientBoostingRegressor` using a Pipeline.
7. **Model Saving**: Save the trained pipeline as `salary_prediction_pipeline.pkl`.

---

## 🧰 Technologies Used

- **Python**: Core programming language.
- **Streamlit**: For building the web application.
- **Scikit-learn**: For ML modeling and preprocessing.
- **Pandas**: For data manipulation.
- **NumPy**: For numerical operations.
- **Matplotlib & Seaborn**: For visualization.
- **Joblib**: For saving/loading models.

---

## 🔧 Setup and Installation

### Step 1: Clone the Repository
```bash
git clone https://github.com/ankushkumar10/Employee_Salary_Prediction.git
cd Employee_Salary_Prediction
````

### Step 2: Create a Virtual Environment (Recommended)

```bash
python -m venv venv
```

#### On Windows:

```bash
.\venv\Scripts\activate
```

#### On macOS/Linux:

```bash
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Prepare Dataset

Make sure you have a `data` folder and `salary_dataset.csv` inside it:

```bash
mkdir data
# Place salary_dataset.csv inside the data folder
```

### Step 5: Train the Model

```bash
python train_model.py
```

This script will save the pipeline as `salary_prediction_pipeline.pkl`.

### Step 6: Run the Streamlit App

```bash
streamlit run app.py
```

The app will open in your default browser.

---

## 💻 Usage

* **Salary Prediction**: Navigate to the "Salary Prediction" page, input details, and click **Predict Salary**.
* **Data Exploration**: Check the "Data Exploration" page for insights and visualizations.
* **About**: Learn about the app and methodology.

---

## 📁 Project Structure

```
.
├── data/
│   └── salary_dataset.csv            # Raw dataset
├── notebook/
│   └── salary_prediction.ipynb       # Jupyter Notebook with EDA
├── app.py                            # Streamlit app
├── train_model.py                    # Model training script
├── requirements.txt                  # Dependencies
├── salary_prediction_pipeline.pkl    # Trained model
└── README.md                         # Project documentation
```

---

## 📊 Data Source

The application uses `salary_dataset.csv`, containing employee attributes and salaries. It's used for both training the model and interactive exploration.

---

## 📈 Results & Insights

The **Gradient Boosting Regressor** outperformed other models like Linear Regression and Random Forest, achieving the best **R² Score**. The data exploration section reveals trends in salary distributions, departmental differences, and correlations.

---

## 🌱 Future Enhancements

* Advanced feature engineering and interactions
* Hyperparameter tuning
* Ensemble modeling techniques
* Cloud deployment (AWS, GCP, Azure)
* Secure user authentication
* Feedback system for continuous improvement

---

## 📝 License

This project is licensed under the **MIT License** – see the [LICENSE.md](LICENSE.md) file for details.
