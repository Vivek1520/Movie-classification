import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm


genre_list = ['action', 'adult', 'adventure', 'animation', 'biography', 'comedy', 'crime', 'documentary', 'family',
              'fantasy', 'game-show', 'history', 'horror', 'music', 'musical', 'mystery', 'news', 'reality-tv',
              'romance', 'sci-fi', 'short', 'sport', 'talk-show', 'thriller', 'war', 'western']

fallback_genre = 'Unknown'


try:
    with tqdm(total=50, desc="Loading Train Data") as pbar:
        train_data = pd.read_csv('train_data.txt', sep=':::', header=None,
                                 names=['SerialNumber', 'MOVIE_NAME', 'GENRE', 'MOVIE_PLOT'], engine='python')
        pbar.update(50)
except Exception as e:
    print(f"Error loading train_data: {e}")
    raise

X_train = train_data['MOVIE_PLOT'].astype(str).apply(lambda doc: doc.lower())
genre_labels = [genre.split(', ') for genre in train_data['GENRE']]
mlb = MultiLabelBinarizer()
y_train = mlb.fit_transform(genre_labels)


tfidf_vectorizer = TfidfVectorizer(max_features=5000)
with tqdm(total=50, desc="Vectorizing Training Data") as pbar:
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    pbar.update(50)


with tqdm(total=50, desc="Training Model") as pbar:
    naive_bayes = MultinomialNB()
    multi_output_classifier = MultiOutputClassifier(naive_bayes)
    multi_output_classifier.fit(X_train_tfidf, y_train)
    pbar.update(50)


try:
    with tqdm(total=50, desc="Loading Test Data") as pbar:
        test_data = pd.read_csv('test_data.txt', sep=':::', header=None,
                                names=['SerialNumber', 'MOVIE_NAME', 'MOVIE_PLOT'], engine='python')
        pbar.update(50)
except Exception as e:
    print(f"Error loading test_data: {e}")
    raise

X_test = test_data['MOVIE_PLOT'].astype(str).apply(lambda doc: doc.lower())


with tqdm(total=50, desc="Vectorizing Test Data") as pbar:
    X_test_tfidf = tfidf_vectorizer.transform(X_test)
    pbar.update(50)


with tqdm(total=50, desc="Predicting on Test Data") as pbar:
    y_pred = multi_output_classifier.predict(X_test_tfidf)
    pbar.update(50)


test_movie_names = test_data['MOVIE_NAME']
predicted_genres = mlb.inverse_transform(y_pred)
test_results = pd.DataFrame({'MOVIE_NAME': test_movie_names, 'PREDICTED_GENRES': predicted_genres})

test_results['PREDICTED_GENRES'] = test_results['PREDICTED_GENRES'].apply(
    lambda genres: [fallback_genre] if len(genres) == 0 else genres)

for _, row in test_results.iterrows():
    movie_name = row['MOVIE_NAME']
    predicted_genres = ', '.join(row['PREDICTED_GENRES'])
    print(f"Movie: {movie_name}\nPredicted Genres: {predicted_genres}\n")


y_train_pred = multi_output_classifier.predict(X_train_tfidf)

accuracy = accuracy_score(y_train, y_train_pred)
precision = precision_score(y_train, y_train_pred, average='micro')
recall = recall_score(y_train, y_train_pred, average='micro')
f1 = f1_score(y_train, y_train_pred, average='micro')

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")