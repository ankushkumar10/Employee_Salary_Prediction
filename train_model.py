import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
import joblib

df = pd.read_csv('data/salary_dataset.csv')

# --- 1. Data Preprocessing ---

# Fill missing 'Performance Score' with the median
df['Performance Score'].fillna(df['Performance Score'].median(), inplace=True)

# Feature Engineering: Extract date components
df['Joining Date'] = pd.to_datetime(df['Joining Date'])
df['Joining_Year'] = df['Joining Date'].dt.year
df['Joining_Month'] = df['Joining Date'].dt.month
df['Joining_Day'] = df['Joining Date'].dt.day

# Drop original and unnecessary columns
df = df.drop(columns=['ID', 'Name', 'Joining Date', 'Status'])

# Define features (X) and target (y)
X = df.drop('Salary', axis=1)
y = df['Salary']

# --- 2. Create a Preprocessing Pipeline ---

# Identify categorical and numerical features
categorical_features = ['Gender', 'Department', 'Location', 'Session']
numerical_features = ['Age', 'Performance Score', 'Experience', 'Joining_Year', 'Joining_Month', 'Joining_Day']

# Create a column transformer for one-hot encoding categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough' # Keep other columns (if any)
)

# --- 3. Model Training ---

# As per the notebook, GradientBoostingRegressor was the best model.
model = GradientBoostingRegressor(random_state=42)

# Create the full pipeline
full_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', model)
])

# Train the model on the entire dataset
full_pipeline.fit(X, y)

# --- 4. Save the Pipeline ---

# Save the entire pipeline (preprocessor + model) to a file
joblib.dump(full_pipeline, 'salary_prediction_pipeline.pkl')

print("Model training complete and pipeline saved as 'salary_prediction_pipeline.pkl'")