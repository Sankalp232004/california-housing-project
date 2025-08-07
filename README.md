Predicting California Housing Prices with Linear Regression
🎯 Project Goal
The objective of this project is to build a multiple linear regression model to predict the median housing value in California districts based on various features from the 1990 census data.

📊 Dataset
This project uses the classic California Housing Prices dataset, which contains data for 20,640 census block groups in California. The dataset includes 9 features such as median income, housing median age, and location (latitude/longitude), along with the target variable, median_house_value.

📋 Project Workflow
The project followed a standard machine learning workflow:

Data Loading & Initial Exploration: Loaded the dataset and performed an initial inspection to understand its structure and identify issues like missing data.

Data Cleaning & Preprocessing: Handled missing values, engineered new features, and encoded categorical data.

Exploratory Data Analysis (EDA): Used visualizations to uncover patterns, identify key predictors, and understand feature relationships.

Feature Selection: Chose the most relevant features for the model based on EDA.

Model Training: Split the data into training and testing sets and trained a LinearRegression model.

Model Evaluation: Evaluated the model's performance on the test set using standard regression metrics.

🧹 Data Cleaning & Preprocessing
The initial inspection revealed that the total_bedrooms column contained 207 missing values. These were imputed using the median value of the column.

Several new features were engineered to provide more context for the model:

rooms_per_household

bedrooms_per_room

population_per_household

The categorical feature, ocean_proximity, was converted into numerical format using one-hot encoding to be used in the model.

📈 Exploratory Data Analysis (EDA)
The EDA revealed several key insights:

The strongest predictor of housing prices is median_income, showing a clear positive linear relationship.

A geographical plot of latitude and longitude showed that housing prices are significantly higher in coastal areas, especially around the San Francisco Bay Area and Southern California.

🤖 Modeling & Evaluation
A multiple linear regression model was trained on the preprocessed data to predict median_house_value. The model's performance was evaluated on the unseen test set, yielding the following results:

R-squared (R²): 0.60

This indicates that the model successfully explains approximately 60% of the variance in California housing prices based on the selected features.

Mean Absolute Error (MAE): $50,888.66

On average, the model's predictions are off by about $50,889.

Root Mean Squared Error (RMSE): $72,668.54

This is another measure of prediction error, which penalizes larger errors more heavily.

🛠️ Tools Used
Language: Python

Libraries:

Pandas & NumPy (Data Manipulation)

Matplotlib & Seaborn (Data Visualization)

Scikit-learn (Model Training & Evaluation)

Environment: Jupyter Notebook
