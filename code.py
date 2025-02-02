import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


data = {
    'Solar_Irradiance': np.random.uniform(200, 1000, 100),  
    'Temperature': np.random.uniform(10, 40, 100),  
    'Humidity': np.random.uniform(20, 80, 100),  
    'Power_Output': np.random.uniform(1, 10, 100)  
}
df = pd.DataFrame(data)


X = df[['Solar_Irradiance', 'Temperature', 'Humidity']]
y = df['Power_Output']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


fig, ax = plt.subplots()
ax.scatter(y_test, y_pred)
ax.set_xlabel("Actual Power Output (kW)")
ax.set_ylabel("Predicted Power Output (kW)")
ax.set_title("Actual vs Predicted Power Output")
plt.show()
