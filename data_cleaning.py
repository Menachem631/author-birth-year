from datasets import load_dataset
from tqdm import tqdm
import numpy as np
import json
import pandas as pd
def load_data(translation_words, path="sedthh/gutenberg_english", streaming=True, count=200):
    """
    Load Gutenberg books

    Parameters
    ------------------
    translation_words: array
        array of words to look for in a prospective book's metadata, which indicate that the book is a translation and should therefore be excluded.
    path: str
        path to stream the books from
    streaming: bool
        whether the dataset connection should be a streaming connection.
    count: int
        how many books the return should contain

    Returns
    -----------------
    sample: array
        array of books

    """
    ds = load_dataset(path, streaming=streaming)
    sample = []
    counter = 0
    for book in tqdm(ds['train']):
        # only include books that include the word Fiction, to exclude poetry etc, and exclude any word indicating a translation
        if all(word not in book['METADATA'] for word in translation_words) and (
                'fiction' in book['METADATA'] or 'Fiction' in book['METADATA']):
            sample.append(book)
            counter += 1
        if (counter) >= count:
            break
    return sample


class Book:
    def __init__(self, book):
        self.metadata = json.loads(book['METADATA'])
        self.title = self.metadata['title']
        self.text = book['TEXT']
        #extract birth year from metadata, if possible
        try:
            self.birth_year = int(self.metadata['authors'][-9:-5])
        except ValueError:
            self.birth_year = np.nan
        #extract author name and normalize case etc, for grouping.
        try:
            author_raw = self.metadata['authors'].split(',')
            if author_raw[1].find('(') >= 0 or author_raw[0].find('(') >= 0:
                self.author = (author_raw[1][:author_raw[1].find('(')] + author_raw[1][author_raw[1].find(')')+1:]).replace('  ', ' ').strip() + ' ' + author_raw[0]
            else:
                self.author = author_raw[1].strip() + ' ' + author_raw[0]
        except IndexError:
            self.author = np.nan
    def __repr__(self):
        return self.title

class Corpus:
    def __init__(self, corpus):
        self.corpus = corpus

    def clean_corpus(self, first_part_only=True, split=20000, remove=3000):
        """
        Clean each book within the corpus and return a dataframe with the texts, author names and birthyears.

        Parameters
        ------------------
        first_part_only: bool
            whether to include only the first segment of a book
        split: int
            how many characters in each segment of a book
        remove: int
            how many characters to remove from the beginning and end of a book, to exclude indices, credits, publishing information etc.

        Returns
        -----------------
        df: pd.DataFrame
            a pandas dataframe with columns containing raw text, author names, book titles, book segment numbers and author birth years.

        """
        texts = []
        authors = []
        birth_years = []
        titles = []
        numbers = []
        for book in self.corpus:
            book = Book(book)
            # remove books that are too short
            if len(book.text) < split + 2 * remove:
                continue
            # extract data from first segment of books
            if first_part_only:
                texts.append(book.text[remove:remove + split])
                authors.append(book.author)
                birth_years.append(book.birth_year)
                titles.append(book.title)
            # extract data from each segment of books
            else:
                for slice_index in range(0, int(np.floor((len(book.text) - 2 * remove) / split))):
                    texts.append(book.text[remove + slice_index * split: remove + (slice_index + 1) * split])
                    authors.append(book.author)
                    birth_years.append(book.birth_year)
                    titles.append(f"{book.title} Part {slice_index + 1}")
                    numbers.append(slice_index + 1)

        # create dataframe, dropping books with invalid author names or author birth years and duplicates
        self.df = pd.DataFrame()
        self.df['text'] = texts
        self.df['author'] = authors
        self.df['title'] = titles
        self.df['number'] = numbers
        self.df['birth_year'] = birth_years
        self.df = self.df.dropna()
        self.df = self.df.drop_duplicates(subset='title')
        self.df = self.df.reset_index(drop=True)

        return self.df