import streamlit as st
import pandas as pd
from joblib import load

model = load('F:/Github/Home-Credit-Scorecard-Model/random_hci.joblib')

st.write("""
# ScoreCard Predict App for Home Credit Indonesia


This is predict data historical client!
""")

st.sidebar.header('Input Data')

def user_input_features():
    AMT_INCOME_TOTAL= st.sidebar.slider('Income Total',  1000, 1000000, 0)
    AMT_ANNUITY	= st.sidebar.slider('Annuity', 1000, 1000000, 0)
    AMT_CREDIT = st.sidebar.slider('Amount Credit', 0, 100, 0)
    AMT_GOODS_PRICE = st.sidebar.slider('Good Price', 0, 1000000, 0)
    CODE_GENDER = st.sidebar.slider('Gender', 0, 1, 0)
    FLAG_WORK_PHONE = st.sidebar.slider('Work Phone', 0, 1, 0)
    CNT_CHILDREN = st.sidebar.slider('Children', 0, 20, 0)
    FLAG_MOBIL = st.sidebar.slider('Flag Car', 0, 1, 0)
    data = {'AMT_INCOME_TOTAL': AMT_INCOME_TOTAL,
        'AMT_ANNUITY' : AMT_ANNUITY,
        'AMT_CREDIT' : AMT_CREDIT,
        'AMT_GOODS_PRICE' :	AMT_GOODS_PRICE,
        'CODE_GENDER' :	CODE_GENDER,
        'FLAG_WORK_PHONE' :	FLAG_WORK_PHONE,
        'CNT_CHILDREN' : CNT_CHILDREN,
        'FLAG_MOBIL' : FLAG_MOBIL}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader ('Input Data')
st.write(df)

prediction = model.predict(df)
st.subheader('Prediction')
st.write('Predict Class:', prediction[0])
