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

def get_row_count_from_folder(path=Path(r"C:\Users\menac\PycharmProjects\Text Analysis\stylo")):
    return sum([int(item.stem.split('_')[-1]) for item in path.iterdir()])

def get_file_count_from_folder(path=Path(r"C:\Users\menac\PycharmProjects\Text Analysis\stylo")):
    return len([1 for item in path.iterdir()])

def stylo(X, partition=False, folder_path=r"C:\Users\menac\PycharmProjects\Text Analysis\stylo", batch_size=32,
          n_batches=100, to_parquet=True):
    folder_path = Path(folder_path)
    to_keep = ['NOUN', 'ADP', 'VERB', 'PRON', 'ADV', 'PART', 'NUM', 'DET',
               'SCONJ', 'INTJ', 'PROPN', 'AUX', 'ADJ', 'CCONJ', ',', '.', '“', '!',
               '”', '(', ')', '?', ':', '—', '-', ';', '‘', '’']
    rows = []

    if (type(X) == pd.Series):
        iterable = X
    elif (type(X) == pd.DataFrame):
        iterable = X['text']

    i = get_row_count_from_folder(folder_path)
    iterable = iterable[i:]

    for doc in tqdm(nlp.pipe(iterable, n_process=1, batch_size=batch_size), total=len(iterable)):
        pos = Counter([token.pos_ for token in doc])
        word_count = len(doc) - pos.get('SPACE', 0) - pos.get('PUNCT', 0) - pos.get('X', 0) - pos.get('SYM', 0)
        unique_count = len(
            set([token.text.lower() for token in doc if token.pos_ not in ['SPACE', 'X', 'PUNCT', 'SYM']]))
        sent_count = sum([1 for _ in doc.sents])
        punct = Counter([token.text for token in doc if token.pos_ == 'PUNCT'])
        raw = dict(pos + punct)
        row = {key: raw[key] / word_count for key in raw.keys() if key in to_keep}
        row['uwc'] = unique_count / word_count
        row['wps'] = word_count / sent_count
        rows.append(row)
        if len(rows) % (batch_size * n_batches) == 0:
            df = pd.DataFrame(rows)
            df = df.fillna(0)
            file_count = get_file_count_from_folder(folder_path)
            if to_parquet:
                df.to_parquet(folder_path / f"{file_count}_{len(rows)}.parquet")
                print(f"wrote {file_count}_{len(rows)}.parquet")
            rows = []

    if len(rows) % batch_size * n_batches != 0:
        df = pd.DataFrame(rows)
        df = df.fillna(0)
        file_count = get_file_count_from_folder(folder_path)
        if to_parquet:
            df.to_parquet(folder_path / f"{file_count}_{len(rows)}.parquet", index=True)
            rows = []

    return df


model = joblib.load("ridge.pkl")

def everything(x_raw, model):
    if len(x_raw)==0:
        return "Waiting for Input"
    if len(x_raw) < 1000:
        return "Please provide a longer input."
    x = pd.Series(x_raw)
    orig_df = stylo(x, to_parquet=False)
    extra_cols = list(set(['NOUN', 'ADP', 'VERB', 'PRON', 'ADV', 'PART', 'NUM', 'DET',
       'INTJ', 'PROPN', 'AUX', 'SCONJ', 'ADJ', 'CCONJ', ',', '.', '“', '!',
       '”', '(', ')', '?', ':', '—', '-', ';', '‘', '’', 'uwc', 'wps']) - set(orig_df.columns))
    extra_df = pd.DataFrame(columns = extra_cols, data=[[0]*len(extra_cols)])
    df = pd.concat([orig_df,extra_df], axis=1).sort_index(axis=1)
    return int(model.predict(df)[0])



st.set_page_config(layout='wide')
st.title("Predict the Birth Year")
r1c1, r1c2, r1c3, r1c4 = st.columns([1, 1, 1, 1])
with r1c1:
    st.write("A histogram-based gradient boosted regressor to predict author's birth year based on text, "
             "using stylometric features and tfidf words and characters.")
with r1c3:
    st.metric('MAE', "25 years")
with r1c4:
    st.metric('R2', "51%")

r2c1, r2c2 = st.columns([1, 1])
with r2c1:
    img = Image.open("hist.png")
    st.image(img,  width = 350)
with r2c2:
    img = Image.open("hist.png")
    st.image(img,  width = 350)

st.divider()

r3c1, r3c2, r3c3 = st.columns([1, 1, 1])
with r3c1:
    input_text = st.text_input("Enter your text...")
    st.button("Submit Text")
    prediction = everything(input_text, model)


with r3c2:
    st.metric("Prediction", prediction)

with r3c3:
    img = Image.open("hist.png")
    st.image(img,  width = 350)