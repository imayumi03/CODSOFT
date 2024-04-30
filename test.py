import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

def load_data(filepath):
    data = []
    with open(filepath, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split(':::')
            if len(parts) >= 3:
                movie_id = parts[0].strip()
                movie_title = parts[1].strip()
                genre = parts[2].strip() if len(parts) > 3 else 'Unknown'
                plot_summary = parts[3].strip() if len(parts) > 3 else parts[2].strip()
                data.append([movie_id, movie_title, genre, plot_summary])
    return pd.DataFrame(data, columns=['movie_id', 'movie_title', 'genre', 'plot_summary'])

# Load the data
train_data = load_data('train_data.txt/train_data.txt')
test_data = load_data('test_data.txt/test_data.txt')

# Initialize the TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english')

# Fit and transform the training data
X_train = tfidf_vectorizer.fit_transform(train_data['plot_summary'])
y_train = train_data['genre']

# Transform the test data
X_test = tfidf_vectorizer.transform(test_data['plot_summary'])

# Classifiers to use
classifiers = {
    'Naive_Bayes': MultinomialNB(),
    'Logistic_Regression': LogisticRegression(max_iter=1000),
    'SVM': SVC(kernel='linear')
}

# Predict and save results for each classifier
for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    test_data['genre'] = predictions
    # Save to CSV
    filename = f'{name}_predicted_test_data.csv'
    test_data.to_csv(filename, index=False)
    print(f'{name} predictions saved to {filename}')

# Optionally, display a summary from one of the files
print(pd.read_csv('Naive_Bayes_predicted_test_data.csv').head())
69