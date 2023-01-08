import pandas as pd
import random
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import *


def init_genres():
    lst = ['Christian', 'Country', 'Pop', 'Rock', 'R&B']
    genres = (random.choice(lst), random.choice(lst))
    print('Ваши жанры:', genres[0], 'и', genres[1])
    return genres


def main():
    data = get_data(input('Введите имя файла с жанрами'))
    genres = init_genres()
    print('Мои жанры:', genres)
    columns = six_paths_of_pain(get_my_genres_data(data, genres))
    x_train, x_test, y_train, y_test = split_data(columns)
    vectorizer, vectorized_x_train = vectorize(x_train)
    clf = MultinomialNB()
    clf.fit(vectorized_x_train, y_train)
    vectorized_x_test = vectorizer.transform(x_test)
    clf.predict(vectorized_x_test)
    pred = clf.predict(vectorized_x_test)
    print(classification_report(y_test, pred))
    my_text(genres, clf, vectorizer)
    last()


def my_text(genres, clf, vectorizer):
    song_1_genre = genres[0]
    song_2_genre = genres[1]
    song_1_lyrics = input(f'Введите текст песни жанра {song_1_genre}')
    song_2_lyrics = input(f'Введите текст песни жанра {song_2_genre}')
    columns = pd.DataFrame(data={"genre": [song_1_genre, song_2_genre], "lyrics": [song_1_lyrics, song_2_lyrics]})
    columns = six_paths_of_pain(columns)
    pred = clf.predict(vectorizer.transform(columns['lemmatized']))
    print(classification_report(columns['genre'], pred))


def last():
    data = get_data(input('Введите имя файла с авторами'))
    columns = data[['cantorNome', 'letra']]
    columns = columns[(columns.cantorNome == "david-bowie") | (columns.cantorNome == "paul-mccartney")]
    columns['lyrics'] = columns['letra']
    columns = six_paths_of_pain(columns)
    x_train, x_test, y_train, y_test = split_data_new(columns)
    vectorizer, vectorized_x_train = vectorize(x_train)
    clf = MultinomialNB()
    clf.fit(vectorized_x_train, y_train)
    vectorized_x_test = vectorizer.transform(x_test)
    clf.predict(vectorized_x_test)
    pred = clf.predict(vectorized_x_test)
    print(classification_report(y_test, pred))


def six_paths_of_pain(columns):
    columns = to_lowercase(columns)
    columns = tokenize(columns)
    columns = stop_del(columns)
    columns = lemmatize(columns)
    return columns


def split_data(columns):
    return train_test_split(columns.lemmatized, columns.genre, train_size=0.7)


def split_data_new(columns):
    return train_test_split(columns.lemmatized, columns.cantorNome, train_size=0.7)


def vectorize(x_train):
    vectorizer = CountVectorizer(ngram_range=(1, 3))
    return vectorizer, vectorizer.fit_transform(x_train)


def stop_del(columns):
    noise = nltk.corpus.stopwords.words('english')
    withoutstop = columns['tokened'].apply(lambda x: [item for item in x if item not in noise])
    without_stop = []
    for a in withoutstop:
        without_stop.append(", ".join(a))
    columns['without_stop'] = without_stop
    return columns


def tokenize(columns):
    tokened = columns.apply(lambda row: nltk.word_tokenize(row['lowered']), axis=1)
    columns['tokened'] = tokened
    return columns


def to_lowercase(columns):
    lowered = columns['lyrics'].str.lower()
    columns['lowered'] = lowered
    return columns


def lemmatize(columns):
    lemmatizer = WordNetLemmatizer()
    lemmatized = columns['without_stop'].apply(lambda x: [lemmatizer.lemmatize(x)])
    lemma = []
    for a in lemmatized:
        lemma.append(", ".join(a))
    columns['lemmatized'] = lemma
    return columns


def get_my_genres_data(data, genres):
    genre1, genre2 = genres
    columns = data[['genre', 'lyrics']]
    columns = columns[(columns.genre == genre1) | (columns.genre == genre2)]
    return columns


def get_data(filename):
    try:
        return pd.read_csv(filename)
    except FileNotFoundError:
        print('Файл не найден')
        exit()


if __name__ == '__main__':
    main()
