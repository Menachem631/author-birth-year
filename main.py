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

densifier = FunctionTransformer(lambda X: X.toarray(), accept_sparse=True)

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

df.drop_duplicates()
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
df = df.drop(columns = ['birthyear'])
df_dropped = df.dropna()
df_dropped = df_dropped[df_dropped['text'].str.len() > 11000].reset_index(drop=True)
text = []
for index, row in df_dropped.iterrows():
    text.append(row.text[3000:13000])
df_dropped['text'] = text
X_raw = df_dropped['text']
y = df_dropped['pub_year']
gss = GroupShuffleSplit(n_splits=1, random_state=42)
train_idx, test_idx = next(gss.split(X_raw, y, groups=df_dropped['author']))
X_train_raw = X_raw[train_idx]
X_test_raw = X_raw[test_idx]
y_train = y[train_idx]
y_test = y[test_idx]
gkf = GroupKFold(n_splits=3)

print(df_dropped.shape)

# baseline_pipe = Pipeline([('tfid', TfidfVectorizer(max_features=2000, stop_words='english')),
#                               ('densifier', densifier), ('model', HistGradientBoostingRegressor())])
#
# scores = cross_validate(baseline_pipe, X_train_raw, y_train, n_jobs=-1, groups=df_dropped['author'][train_idx],
#                                 scoring={'mae': 'neg_mean_absolute_error', 'r2': 'r2'}, cv=gkf, return_train_score=True)
#
# print(scores)
#
with mlflow.start_run(run_name='experiment1'):

    baseline_pipe = Pipeline([('tfid', TfidfVectorizer(max_features=2000, stop_words='english')),
                              ('densifier', densifier), ('model', HistGradientBoostingRegressor(l2_regularization=1))])

    scores = cross_validate(baseline_pipe, X_train_raw, y_train, n_jobs=-1, groups=df_dropped['author'][train_idx],
                                scoring={'mae': 'neg_mean_absolute_error', 'r2': 'r2'}, cv=gkf, return_train_score=True)

    mlflow.log_metric('mae', -scores['test_mae'].mean())
    mlflow.log_metric('train_mae', -scores['train_mae'].mean())
    mlflow.log_metric('r2', scores['test_r2'].mean())
    mlflow.log_metric('train_r2', scores['train_r2'].mean())
#
    print(scores)




    def vocab_counter(text):
        words = word_tokenize(text)
        return len(set(words)) / len(words)


    def sent_length(text):
        words = word_tokenize(text)
        sent = sent_tokenize(text)
        return len(words) / len(sent)





    class StylometricFeatures(BaseEstimator, TransformerMixin):
        def fit(self, X, y=None):
            return self

        def transform(self, X):
            features = []
            for text in X:
                words = word_tokenize(text)
                sent = sent_tokenize(text)
                vc = len(set(words)) / len(words)
                sl = len(words) / len(sent)
                awl = np.sum([len(word) for word in words if word.isalpha()]) / len([w for w in words if w.isalpha()])
                scr = len([word for word in words if word == ';']) / len(words)

                features.append({'vc': vc, 'sl': sl, 'awl': awl, 'scr': scr})
            return csr_matrix(pd.DataFrame(features).values)

        def get_feature_names_out(self, input_features=None):
            return np.array(['vocab', 'sentence_length', 'word_length', 'semicolon_usage'])

with mlflow.start_run(run_name='experiment2'):


    def objective(trial):
        params = {'l2_regularization': trial.suggest_float('l2_regularization', 1e-1, 100, log=True),
                  'learning_rate': trial.suggest_float('learning_rate', .05, 0.2),
                  'max_iter': trial.suggest_int('max_iter', 50, 300),
                  'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 15, 63)}

        pipe = Pipeline([('features', FeatureUnion(
            [('stylo', StylometricFeatures()), ('tfidf', TfidfVectorizer(max_features=2000, stop_words='english'))])),
                         ('densifier', densifier), ('model', HistGradientBoostingRegressor(**params, early_stopping=True, n_iter_no_change=10))])
        scores = cross_validate(pipe, X_train_raw, y_train, groups=df_dropped['author'][train_idx],
                                scoring={'mae': 'neg_mean_absolute_error', 'r2': 'r2'}, cv=gkf, return_train_score=True)

        print(scores)
        return -scores['test_mae'].mean()


    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=20, timeout=1200)
    print(study.best_params)
    mlflow.log_metric('mae', study.best_value)
    mlflow.log_params(study.best_params)
    print(study.best_value)

    ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)

    #
    # search = GridSearchCV(estimator=pipe, n_jobs=-1, return_train_score=True, scoring={'mae': 'neg_mean_absolute_error', 'r2': 'r2'}, refit='mae', cv=gkf, param_grid={'model__max_depth': [None, 1, 2, 3],
    #                                                                                                                                                                    'model__l2_regularization': [0, 1, 10],
    #                                                                                                                                                                    'model__learning_rate': [0.01, 0.1],
    #                                                                                                                                                                    'model__max_iter': [50, 100],
    #
    #                                                                                                                                                                    'model__max_leaf_nodes': [7, 15]})
    # search.fit(X_train_raw, y_train, groups=df_dropped['author'][train_idx])
    # # pred = pipe.predict(X_test_raw)
    # # r2_score = r2_score(y_test, pred)
    # # mae = mean_absolute_error(y_test, pred)



    # mlflow.log_params(search.best_params_)
    # mlflow.log_metric('mae', -search.cv_results_['mean_test_mae'][search.best_index_])
    # mlflow.log_metric('r2', search.cv_results_['mean_test_r2'][search.best_index_])
    # mlflow.log_metric('train_mae', -search.cv_results_['mean_train_mae'][search.best_index_])
    # mlflow.log_metric('train_r2', search.cv_results_['mean_train_r2'][search.best_index_])

    # print(search.cv_results_)



    # mlflow.log_metrics(metrics={'r2': r2_score, 'mae': mae})
    # mlflow.log_param('max_feat', 2000)
    # mlflow.sklearn.log_model(pipe, 'pipeline')

    # import shap
    # features = pipe.named_steps['features']
    # model = pipe.named_steps['model']
    # X_te = features.transform(X_test_raw).toarray()
    # names = pipe.named_steps['features'].get_feature_names_out()
    # explainer = shap.TreeExplainer(model)
    # shap_values = explainer.shap_values(X_te)
    # shap.summary_plot(shap_values, X_te, names)