
🧠 NYC Taxi Trip Prediction
Fare Amount Prediction & Peak-Hour Classification (Machine Learning Project)

This project analyzes NYC Yellow Taxi data to predict trip fare amounts and classify whether a trip occurs during peak hours.
It includes end-to-end preprocessing, feature engineering, training, evaluation, and exporting models for future use.

📌 Project Objectives
1. Predict the trip fare (fare_amount)

Using a regression model based on:

Distance (pure distance, trip_distance)

Trip duration (log-trip-duration)

Pickup/dropoff location

Time-related features (night, weekend, hour, day)

2. Classify “Peak vs Non-Peak” trips

Binary classification model to identify congestion/high-demand periods.

📂 Project Structure
├── taxiPrediction.ipynb        # Main Jupyter Notebook
├── peak_classification_model.pkl   # Saved classification model
├── fare_prediction_model.pkl       # (optional) Saved regression model
├── preprocessor.pkl                # Scaling/encoding pipeline
├── README.md                       # Project documentation

🧼 Data Preprocessing

The dataset originally included 166,067 rows (downsampled for speed).
Main preprocessing steps:

✔️ Missing value handling

Rows with impossible values (zero distance, zero fare, negative duration) were removed.

✔️ Feature Engineering

Pure distance (Haversine)

Log-trip-duration (stable, less skewed)

Night/weekend indicators

Pickup hour, day of week, day of month

Airport pickup/dropoff flags

Manhattan zone flags

Congestion fee indicators

✔️ Scaling

Sensitive numeric variables were standardized:

distance_pure

trip_duration

Geographic coordinates

📊 Correlation Analysis

Strong positive correlations with fare:

distance_pure (0.84)

trip_distance (0.69)

log_trip_duration (0.64)

PU_longitude (0.58)

Strong negative correlations:

is_same_borough (-0.70)

is_manhattan_pickup (-0.60)

PU_latitude (-0.43)

All revenue-related columns (tips, tolls, extra fees) were excluded in order to build a more realistic predictive model.

🤖 Machine Learning Models
Fare Prediction (Regression)

LightGBM Regressor

RandomForestRegressor

Hyperparameter tuning:
Performed using RandomizedSearchCV to reduce time while exploring wide parameter spaces.

Peak Classification (Binary Classification)

Model: RandomForestClassifier

Saved using joblib:

joblib.dump(modelClassification, "peak_classification_model.pkl")


Loaded later instantly (no retraining):

modelClassification = joblib.load("peak_classification_model.pkl")

🧪 Model Evaluation
Fare prediction metrics

RMSE

MAE

R² Score

Error distribution plots

True vs Predicted scatter plot

Peak classification metrics

Accuracy

Precision

Recall

F1 score

Confusion Matrix

ROC-AUC Curve

Example classification results:

precision    recall  f1-score   support
0       1.00      1.00      1.00    296908
1       1.00      1.00      1.00    261128

💾 Saving and Loading Models
Save model:
joblib.dump(model, "model_name.pkl")

Load model later (no training needed):
model = joblib.load("model_name.pkl")

📈 Visualization

The project includes:

Correlation heatmaps

Feature importance plots

Geographic maps (pickup/dropoff distributions)

Distribution plots (distance, duration, fare)

Barplots of selected correlations (distance & duration features)

Example correlation barplot code:

plt.figure(figsize=(6,4))
sns.barplot(x=high_corr_features.values, y=high_corr_features.index, palette="viridis")
plt.title("Selected correlations with fare_amount")
plt.show()

🚀 How to Run the Project
1. Install dependencies
pip install -r requirements.txt

2. Open the Notebook
jupyter notebook taxiPrediction.ipynb

3. (Optional) Use saved models
model = joblib.load("peak_classification_model.pkl")
pred = model.predict(X_test)

📜 License

This project is open-source and available for educational and research purposes.
