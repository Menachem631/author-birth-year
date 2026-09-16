from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from tqdm import tqdm
from collections import Counter
from scipy.sparse import csr_matrix
import spacy
import re
import numpy as np
from joblib import Memory
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner', 'lemmatizer'])
nlp.add_pipe('sentencizer')

location = "./cachedir"
memory = Memory(location, verbose=1)


@memory.cache
def stylo_chunk(X):
    to_keep = ['NOUN', 'ADP', 'VERB', 'jON', 'ADV', 'PART', 'NUM', 'DET',
               'SCONJ', 'INTJ', 'PROPN', 'AUX', 'ADJ', 'CCONJ', ',', '.', '“', '!',
               '”', '(', ')', '?', ':', '—', '-', ';', '‘', '’']
    rows = []

    if (type(X) == pd.Series):
        iterable = X
    elif (type(X) == pd.DataFrame):
        iterable = X['text']


    for doc in tqdm(nlp.pipe(iterable, n_process=1, batch_size=32), total=len(iterable)):
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
        df = pd.DataFrame(rows)
        add_columns = set(to_keep) - set(df.columns)
        for col in add_columns:
            df[col] = np.nan
        df = df.fillna(0)
        df.sort_index(axis=1, inplace=True)
    return df

def stylo(X, chunksize = 320):
    chunks = np.array_split(X, np.ceil(len(X)/chunksize))
    results = pd.DataFrame()
    for chunk in chunks:
        new = stylo_chunk(chunk)
        results = pd.concat([results, new])
    return results

class CleanText(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        texts = []
        if (type(X) == pd.Series):
            iterable = X
        elif (type(X) == pd.DataFrame):
            iterable = X['text']
        for text in iterable:
            text = text.replace('\r\n', '\n', )
            text = text.replace('[Illustration]', '')
            text = text.replace('cover\n', '\n')
            text = re.sub(r' {2,}', ' ', text)
            text = re.sub(r'\n{3,}', '\n\n', text)
            text = text.replace('\xa0', '')
            text = text.strip()
            texts.append(text)
        if (type(X) == pd.Series):
            return pd.Series(texts)
        elif (type(X) == pd.DataFrame):
            X['text'] = texts
            return X


class StylometricFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = stylo(X)
        self.feature_names = df.columns
        return csr_matrix(df.values)

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names)