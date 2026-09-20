import streamlit as st
from PIL import Image
import pandas as pd
from pathlib import Path
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
    shap.waterfall_plot(shap_values[0], max_display=7, show=False)

@st.cache_resource
def display_books_dist():
    img = Image.open(r"charts/books_by_author.png")
    return img

@st.cache_resource
def display_shap_training():
    img = Image.open("charts/shap.png")
    return img


st.set_page_config(layout='wide')
st.title("Predict Author Birth Year From Text")
r1c1, _1, r1c2, r1c3, _2 , r1c4 = st.columns([1.2, 0.2, 0.8, 1.2, 0.2, 1.2])
with r1c1:
    st.write("A histogram-based gradient boosted regressor to predict author's birth year based on text, "
             "using stylometric features and tfidf words and characters, utilizing Bayesian Hyperparameter Tuning (Optuna). Dataset was 2000 books "
             "and 461 authors from Project Gutenberg. Included is a Shap analysis, "
             "both model-wide and at text input level, as well as 80% confidence intervals "
             "using Conformalized Quantile Regression.  Try it with your own text down below, or use the demo.")
with r1c2:
    st.metric('MAE', "14.83 years", '9.53 years better than median')
    st.metric('R$^2$', "50.4%")
    st.metric('CQR 80% PI Coverage', "73.4%")
    st.caption('Metrics computed on GroupKFold on author')
with r1c3:
    img = display_books_dist()
    st.image(img,  width = 350)
with r1c4:
    img = Image.open("charts/shap.png")
    st.image(img, width = 350, output_format='PNG')


st.divider()

r3c1, r3c2, r3c3 = st.columns([1, 1, 1])


with r3c1:
    input_text = st.text_input("Enter your text.")
    st.button("Submit Text", type='primary')

    try_finley = st.button('Try with Elsie At Nantucket, by Martha Finley (1827-1909)', type='secondary')
    if try_finley:
        with open(r"example_texts/martha_finley_elsie_nantucket_ch7.txt", 'r') as f:
            input_text = f.read()



with st.spinner('Processing'):
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

