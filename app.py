import streamlit as st
from PIL import Image
import pandas as pd
from pathlib import Path
from collections import Counter
from tqdm import tqdm
import spacy
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner', 'lemmatizer'])
nlp.add_pipe('sentencizer')
import joblib
import shap
import numpy as np
import matplotlib.pyplot as plt

def get_row_count_from_folder(path=Path(r"C:\Users\menac\PycharmProjects\Text Analysis\stylo")):
    return sum([int(item.stem.split('_')[-1]) for item in path.iterdir()])

data_prep = joblib.load('data_prep.pkl')
names_raw = data_prep.named_steps['columns'].named_transformers_['features'].get_feature_names_out()
names = [name.replace('stylo__', '').replace('tfidf_char__', '').replace('tfidf__', '') for name in names_raw]
mqr = joblib.load('mqr.pkl')
median_model = mqr.estimators_[2]
explainer = shap.TreeExplainer(median_model, feature_names = names)
display_scaler = joblib.load('display_scaler.pkl')

def prep_input(x_raw):
    if len(x_raw)==0:
        return "Waiting for input"
    if len(x_raw) < 1000:
        return "Min length is 1000 characters."

    features = data_prep.transform(pd.DataFrame([x_raw], columns=['text']))

    return features

def predict(features):
    print(features)
    pred_raw, pi = mqr.predict(features)
    pred = pred_raw[0]
    low, high = pi[0][0][0], pi[0][1][0]
    return (pred, low, high)


def show_shap(features):
    shap_values = explainer(features)
    shap_values.display_data = display_scaler.transform(features)
    shap.waterfall_plot(shap_values[0], show=False)


st.set_page_config(layout='wide')
st.title("Predict Author Birth Year From Text Using Machine Learning")
r1c1, r1c2, r1c3, r1c4 = st.columns([1, 1, 1, 1])
with r1c1:
    st.write("A histogram-based gradient boosted regressor to predict author's birth year based on text, "
             "using stylometric features and tfidf words and characters. Included is a Shap analysis, "
             "both model-wide and at text input level, as well as 80% confidence intervals "
             "using Conformalized Quantile Regression.")
with r1c3:
    st.metric('MAE', "14.83 years")
with r1c4:
    st.metric('R2', "50.4%")

r2c1, r2c2 = st.columns([1, 1])
with r2c1:
    img = Image.open(r"charts/books_by_author.png")
    st.image(img,  width = 350)
with r2c2:
    img = Image.open("charts/shap.png")
    st.image(img,  width = 350)

st.divider()

d1, d2, d3 = st.columns([1, 1, 1])
r3c1=d1.empty()
r3c2=d2.empty()
r3c3=d3.empty()


@st.fragment
def run_dynamic():
    with r3c1.container():
        input_text = st.text_input("Enter your text...")
        st.button("Submit Text")
        features = prep_input(input_text)


    with r3c2:
        if input_text:
            if type(features) == np.ndarray:
                pred, low, high = predict(features)
                st.metric("Prediction", f"{round(pred)} ({round(low)} - {round(high)})")
            else:
                st.metric("Prediction", features)


    with r3c3:
        if input_text:
            if type(features) == np.ndarray:
                figure = plt.figure()
                show_shap(features)
                st.pyplot(figure)

run_dynamic()