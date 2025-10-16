import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from fastapi import FastAPI
import mlflow
import uvicorn

# Load dataset (inline for demo)
df = pd.DataFrame({
    "sqft": [1000, 1500, 2000, 2500, 3000],
    "bedrooms": [2, 3, 3, 4, 4],
    "age": [10, 5, 20, 15, 8],
    "price": [200000, 250000, 300000, 350000, 400000]
})

X = df.drop("price", axis=1)
y = df["price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# MLflow tracking
mlflow.set_experiment("house_price_demo")
with mlflow.start_run():
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds, squared=False)
    mlflow.log_metric("rmse", rmse)
    mlflow.sklearn.log_model(model, "model")

# FastAPI app
app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "House Price Prediction API"}

@app.get("/predict")
def predict(sqft: float, bedrooms: int, age: int):
    input_data = np.array([[sqft, bedrooms, age]])
    prediction = model.predict(input_data)[0]
    return {"predicted_price": round(prediction, 2)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
