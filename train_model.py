import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import pickle

#load dataset
df = pd.read_csv("sales data.csv", encoding="latin1")

#take useful columns from dataset
df = df[[
    "Order Date",
    "Category",
    "Sub-Category",
    "Region",
    "Segment",
    "Quantity",
    "Discount",
    "Sales"
]]

#convert date
df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")
df["Month"] = df["Order Date"].dt.month
df["Year"] = df["Order Date"].dt.year
df = df.dropna()

#encode categorical columns
encoders = {}
for col in ["Category","Sub-Category","Region","Segment"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

#feature and label
X = df[[
    "Month","Year","Category","Sub-Category",
    "Region","Segment","Quantity","Discount"
]]
y = df["Sales"]

# Train model
model = RandomForestRegressor(n_estimators=100)
model.fit(X, y)

#save model and encoder
pickle.dump(model, open("sales_model.pkl","wb"))
pickle.dump(encoders, open("encoders.pkl","wb"))

print("✅ Advanced model trained")