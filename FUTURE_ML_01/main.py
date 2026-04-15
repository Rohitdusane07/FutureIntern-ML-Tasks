import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# -------------------------------
# 1. Load Data (fix encoding issue)
# -------------------------------
df = pd.read_csv("sales.csv", encoding='latin1')

# -------------------------------
# 2. Data Cleaning
# -------------------------------
df.dropna(inplace=True)
df['Order Date'] = pd.to_datetime(df['Order Date'])
df = df.sort_values('Order Date')

# -------------------------------
# 3. Monthly Sales Graph
# -------------------------------
df['month_year'] = df['Order Date'].dt.to_period('M')

monthly_sales = df.groupby('month_year')['Sales'].sum().reset_index()
monthly_sales['month_year'] = monthly_sales['month_year'].astype(str)

plt.figure()
plt.plot(monthly_sales['month_year'], monthly_sales['Sales'])
plt.xticks(rotation=45)
plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.tight_layout()
plt.show()

# -------------------------------
# 4. Daily Sales Aggregation
# -------------------------------
daily_sales = df.groupby('Order Date')['Sales'].sum().reset_index()

# -------------------------------
# 5. Rolling Average (Trend)
# -------------------------------
daily_sales['rolling_avg'] = daily_sales['Sales'].rolling(window=7).mean()

plt.figure()
plt.plot(daily_sales['Order Date'], daily_sales['Sales'], label="Actual")
plt.plot(daily_sales['Order Date'], daily_sales['rolling_avg'], label="7-Day Avg")
plt.title("Sales Trend with Moving Average")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.xticks(rotation=45)
plt.show()

# -------------------------------
# 6. Prepare Data for Model
# -------------------------------
daily_sales['days'] = (daily_sales['Order Date'] - daily_sales['Order Date'].min()).dt.days

X = daily_sales[['days']]
y = daily_sales['Sales']

# -------------------------------
# 7. Train Model
# -------------------------------
model = LinearRegression()
model.fit(X, y)

# -------------------------------
# 8. Predict Future (30 days)
# -------------------------------
future_days = pd.DataFrame({'days': range(len(X), len(X) + 30)})
predictions = model.predict(future_days)

# -------------------------------
# 9. Forecast Visualization
# -------------------------------
plt.figure()
plt.plot(daily_sales['days'], y, label="Actual Sales")
plt.plot(future_days['days'], predictions, linestyle='dashed', label="Forecast")
plt.title("Sales Forecast (Actual vs Predicted)")
plt.xlabel("Days")
plt.ylabel("Sales")
plt.legend()
plt.grid()
plt.show()

# -------------------------------
# 10. Model Evaluation
# -------------------------------
y_pred = model.predict(X)
mae = mean_absolute_error(y, y_pred)

print("Mean Absolute Error (MAE):", mae)