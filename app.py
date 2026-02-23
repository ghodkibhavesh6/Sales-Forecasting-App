import streamlit as st
import pickle
import pandas as pd

st.title("📊 Smart Sales Prediction System")

#load model and encoder
model = pickle.load(open("sales_model.pkl","rb"))
encoders = pickle.load(open("encoders.pkl","rb"))

#take input from user
category = st.selectbox("Category", encoders["Category"].classes_)
subcat = st.selectbox("Sub-Category", encoders["Sub-Category"].classes_)
region = st.selectbox("Region", encoders["Region"].classes_)
segment = st.selectbox("Segment", encoders["Segment"].classes_)

col1, col2 = st.columns(2)
month = col1.slider("Month",1,12)
year = col2.number_input("Year",2014,2035)

col3, col4 = st.columns(2)
quantity = col3.number_input("Quantity",1,20)
discount = col4.slider("Discount",0.0,1.0)

#convert input in o and 1
cat = encoders["Category"].transform([category])[0]
sub = encoders["Sub-Category"].transform([subcat])[0]
reg = encoders["Region"].transform([region])[0]
seg = encoders["Segment"].transform([segment])[0]

#predict
if st.button("Predict Sales"):
    input_data = [[month,year,cat,sub,reg,seg,quantity,discount]]
    pred = model.predict(input_data)[0]
    st.success(f"Predicted Sales: {pred:,.2f}")