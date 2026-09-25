A machine learning app to estimate a fiction author's birth year based on their work. 

Gutenberg book data -> train, validation, test split based on authors -> data cleaning  -> stylometric features and 
tfidf on words and characters-> gradient boosting regressor -> Optuna hyperparameter tuning ->
quantile loss-based confidence intervals -> Conformalized Quantile Regression -> SHAP Analysis -> 
Streamlit app.

https://textbirthyear.streamlit.app/

Key Results:

R2 = 0.505.

MAE = 14.83 years, down from 24.36 from a naive median prediction.

80% prediction intervals with Conformalized Quantile Regression achieve only 73.4% coverage, 
reaching 100% in the central band of data, but approaching 0% coverage at the data edges.

The most predictive features, corpus-wide, are frequent use of the semi-colon and the comma, 
both indicating early birth years, and frequent use of the period and the word 'big', 
both indicating late birth years.