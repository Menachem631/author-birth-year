from SPARQLWrapper import SPARQLWrapper, JSON


import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import ctypes
ctypes.windll.kernel32.SetThreadExecutionState(0x80000002)
import json
import optuna
import re
from datasets import load_dataset
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.ensemble import HistGradientBoostingRegressor
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
import mlflow
import mlflow.sklearn
from sklearn.model_selection import GroupKFold, RandomizedSearchCV, GridSearchCV, cross_validate
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.sparse import csr_matrix
from sklearn.preprocessing import FunctionTransformer
from nltk import word_tokenize, sent_tokenize

trans_words = ['Translator', 'France', 'French', 'German', 'Russia', 'Spain', 'Sweden', 'Swedish', 'Poland', 'Polish']
all_trans_words = []
for word in trans_words:
    all_trans_words.append(word)
    all_trans_words.append(word.lower())

ds = load_dataset("sedthh/gutenberg_english", streaming=True)
sample = []
counter = 0
for book in tqdm(ds['train']):
    counter += 1
    if all(word not in book['METADATA'] for word in all_trans_words):
        if 'fiction' in book['METADATA'] or 'Fiction' in book['METADATA']:
            sample.append(book)
    if (counter) >= 5000:
        break


def clean_text(text):
    text = text.replace('\r\n', '\n', )
    text = text.replace('[Illustration]', '')
    text = text.replace('cover\n', '\n')
    text = re.sub(r' {2,}', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = text.replace('\xa0', '')
    text = text.strip()
    return text


cleaned = [
    clean_text(book['TEXT']) for book in sample]

birth_years = []
authors = []
authors_lookup = {'Charlotte M. Yonge': 'Charlotte Mary Yonge',
                  'Elizabeth Cleghorn Gaskell': 'Elizabeth Gaskell',
                  'Kate Douglas Smith Wiggin': 'Kate Douglas Wiggin',
                  'Kathleen Thompson Norris': 'Kathleen Norris',
                  'L. M. Montgomery': 'Lucy Maud Montgomery',
                  'T. S. Arthur': 'Timothy Shay Arthur'}
titles = []
publication_years = []
for book in sample:
    book_dict = json.loads(book['METADATA'])
    try:
        birth_years.append(int(book_dict['authors'][-9:-5]))


    except ValueError:
        birth_years.append(np.nan)

    try:
        author_raw = book_dict['authors'].split(',')


        if author_raw[1].find('(') >= 0 or author_raw[0].find('(') >= 0:
            author_cleaned = (author_raw[1][:author_raw[1].find('(')] + author_raw[1][author_raw[1].find(')')+1:]).replace('  ', ' ').strip() + ' ' + author_raw[0]
        else:

            author_cleaned = author_raw[1].strip() + ' ' + author_raw[0]
    except IndexError:
        author_cleaned = np.nan

    if author_cleaned in authors_lookup.keys():
        author_cleaned = authors_lookup[author_cleaned]

    authors.append(author_cleaned)

    try:

        title = book_dict['title']
        print(title)
        if title.find('; Or') >= 0:
            title = title[:title.find('; Or')]
        elif title.find('; or') >= 0:
            title = title[:title.find('; or')]


        elif title.find(' — Complete')>=0:
            title = title[:title.find(' — Complete')]
        elif title.find(', Complete')>=0:
            title = title[:title.find(', Complete')]
        titles.append(title)

    except Exception:
        titles.append(np.nan)

print(authors)
df = pd.DataFrame(columns=['text'], data=cleaned)
df['birthyear'] = birth_years
df['author'] = authors
df['title'] = titles


#
user_agent = 'TextAnalysis/1.0 (https://github.com/Menachem631; jarredbowen770@gmail.com)'



sparql = SPARQLWrapper("https://query.wikidata.org/sparql", agent=user_agent)
sparql.setReturnFormat(JSON)
successes = 0
failures = 0
for i, row in df.iterrows():
    sparql.setQuery(f"""select (min(?pubDate) as ?firstPubDate)
    WHERE {{
      ?work rdfs:label "{row.title}"@en .
      ?work wdt:P577 ?pubDate .
      ?work wdt:P50 ?author .
      ?author rdfs:label "{row.author}"@en .
    }}""")



    try:
        results = sparql.query().convert()
    except Exception:
        failures += 1
        publication_years.append(np.nan)
        continue
    try:
        pub_year = float(results['results']['bindings'][0]['firstPubDate']['value'][:4])
        successes += 1
        publication_years.append(pub_year)
    except Exception:
        failures += 1
        publication_years.append(np.nan)
df['pub_year'] = publication_years

df[['birthyear', 'author', 'title', 'pub_year']].to_csv('harry.csv')

