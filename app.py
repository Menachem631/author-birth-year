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

st.markdown("""
<style>
.block-container{
padding-top:1.5rem
</style>""", unsafe_allow_html=True)

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

@st.cache_resource
def display_pred_vs_actual():
    img = Image.open("charts/actual_vs_pred.png")
    return img

@st.cache_resource
def display_pi_width():
    img = Image.open("charts/pi_width.png")
    return img

@st.cache_resource
def display_error_era():
    img = Image.open("charts/error_by_era.png")
    return img

@st.cache_resource
def display_mae_era():
    img = Image.open("charts/mae_by_era.png")
    return img

@st.cache_resource
def display_pi_era():
    img = Image.open("charts/pi_coverage_by_era.png")
    return img

st.set_page_config(layout='wide')
st.title("Predict Author Birth Year From Text")
r1c1, r1c2, r1c3, r1c4 = st.columns([1.2, 0.8, 1,  3])
with r1c1:
    st.header("Summary")
    st.write("We predict birth year of author based on the text. Since writing style is largely set early on in an author's life, with "
             "changes later being relatively minor, we chose to predict birth year and not year of publication. \n\n"
             "Try it with your own text down below, or use a demo.")

with r1c2:
    st.header("Details")
    with st.popover("Source Data"):
        st.write("Source Data was 2000 books and 461 authors from Project Gutenberg. The book authors have birth years "
                 "spanning from 1628 to 1916, with the majority (80%) between 1809 and 1876.\n\nThe examples of Finley and Barnes "
                 "are from Elsie At Nantucket, Chapter 7, by Martha Finley (The Literature Network) and Gerania by "
                 "Joshua Barnes (University of Michigan), respectively. Neither author was present in the original "
                 "Gutenberg dataset.")
    with st.popover("Feature Engineering"):
        st.write("Books were split into parts, of approximately 20,000 characters each to enhance available training "
                 "data. Books were then split into train, calibration, validation and test sets, ensuring no author "
                 "appeared in more than one set. Features used included stylometric features of punctuation and parts"
                 " of speech, as well as TFIDF on words and characters. Bayesian Hyperparameter Tuning was implemented.")
    with st.popover("Model Details"):
        st.write("A histogram-based gradient boosted regressor was fit, with a quantile loss of 50%, as well as two more "
                 "such models with quantile losses of 10% and 90% to provide the prediction intervals. Conformalized "
                 "Quantile Regression was then applied to the prediction intervals. Shap Analysis is provided, both "
                 "model-wide and at the prediction level.")
    with st.popover("Results Analysis"):
        st.write("The model displays lowest error within the central band of data, between 1809 and 1876, with accuracy "
                 "declining outside that scope, with the model displaying regression towards the mean, with the result that "
                 "very early years are overestimated and very late years are underestimated. Prediction intervals are "
                 "widest for the earliest years.\n\n The most predictive features, corpus-wide, are frequent use of the "
                 "semi-colon and the comma, both indicating early birth years, and frequent use of the period and the "
                 "word 'big', both indicating late birth years.")
    with st.popover("Tech Stack"):
        st.write("spaCy for stylometric features, pandas for data cleaning, matplotlib and seaborn for plotting, "
                 "scikit-learn for feature engineering and model training, Optuna for hyperparameter tuning, SHAP for "
                 "SHAP analysis, MAPIE for Conformalized Quantile Regression and MLflow for experiment tracking")

with r1c3:
    st.header("Metrics")
    st.metric('MAE', "14.83 years", '9.53 years better than median')
    st.metric('R$^2$', "50.4%")
    st.metric('CQR 80% PI Coverage', "73.4%")
    st.caption('Metrics computed on test set of 77 authors born between 1737 and 1892. No author leakage between train and test sets.')

with r1c4:
    st.header("Charts")
    options = ["Shap", "Training Dist.", "Pred Vs Actual", "Error By Era", "MAE By Era", "PI Width", "PI By Era"]
    selection = st.segmented_control(
        "", options, default='Shap'
    )
    if selection == 'Shap':
        img = display_shap_training()
        st.image(img, width = 400, output_format='PNG')
    elif selection == 'Training Dist.':
        img = display_books_dist()
        st.image(img, width=550)
    elif selection == 'Pred Vs Actual':
        img = display_pred_vs_actual()
        st.image(img, width=550, output_format='PNG')
    elif selection == 'Error By Era':
        img = display_error_era()
        st.image(img, width=550, output_format='PNG')
    elif selection == 'MAE By Era':
        img = display_mae_era()
        st.image(img, width=550, output_format='PNG')
    elif selection == 'PI Width':
        img = display_pi_width()
        st.image(img, width=550, output_format='PNG')
    elif selection == 'PI By Era':
        img = display_pi_era()
        st.image(img, width=550, output_format='PNG')



st.divider()

r3c1, r3c2, r3c3 = st.columns([1, 1, 1])


with r3c1:
    input_text = st.text_area("Enter your text. (Minimum 1000 characters)", height=13)
    a1,a2,a3 = st.columns([1,1,1])
    with a1:
        st.button("Submit Text", type='primary')


    with a2:
        try_finley = st.button('Try Finley (b. 1827)', type='secondary')
        if try_finley:
            with open(r"example_texts/martha_finley_elsie_nantucket_ch7.txt", 'r') as f:
                input_text = f.read()
    with a3:
        try_barnes = st.button('Try Barnes (b. 1654)', type='secondary')
        if try_barnes:
            with open(r"example_texts/joshua_barnes_gerania.txt", 'r') as f:
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

            if try_barnes:
                st.write("❌ 1654 is not within (1759-1827)")

            elif try_finley:
                st.write("✅ 1827 is within (1807-1861)")



    with r3c3:
        if input_text:
            if type(features) == np.ndarray:
                figure = plt.figure()
                show_shap(features)
                st.pyplot(figure)


