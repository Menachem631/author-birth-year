from collections import Counter
import pandas as pd
a = [1,2,1,3,4,1,2]
b=[1,3,2,1,4,2]
a_c = Counter(a)
b_c = Counter(b)
df = pd.DataFrame(columns = [1,2,3,4])
df[len(df)+1]= a_c
df[len(df)+1]= b_c
print(df)

import spacy
nlp = spacy.load('en_core_web_sm')
alpha = 'Harry went for a brisk walk.'
beta = 'Even though I was going, I had nevertheless eaten spectacularly, unless you consider that a lie.'
a = Counter([token.pos_ for token in nlp(alpha)])
print(a)
