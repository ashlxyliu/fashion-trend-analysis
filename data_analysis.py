import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import unicodedata

nltk.download('punkt')
nltk.download('stopwords')

data = pd.read_csv('reddit_posts.csv', encoding='utf-8')

def preprocess_text(text):
    """
    Cleans and tokenizes text data.
    """
    try:
        if pd.isna(text) or not isinstance(text, str) or text.strip() == "":
            return ""

        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
        stop_words = set(stopwords.words('english'))
        words = word_tokenize(text.lower())
        words = [word for word in words if word.isalpha() and word not in stop_words]
        return ' '.join(words)
    except Exception as e:
        print(f"Error processing text: {text}, Error: {e}")
        return ""

data['cleaned_title'] = data['title'].apply(preprocess_text)


# Label target: Define a "trendy" post (e.g., score > threshold)
TRENDY_THRESHOLD = 50
data['is_trendy'] = (data['score'] > TRENDY_THRESHOLD).astype(int)

tfidf = TfidfVectorizer(max_features=500, stop_words='english')
tfidf_features = tfidf.fit_transform(data['cleaned_title'].fillna(''))

numerical_features = data[['num_comments', 'upvote_ratio', 'is_self']].fillna(0).reset_index(drop=True)
tfidf_features_df = pd.DataFrame(tfidf_features.toarray())

tfidf_features_df.columns = [f'tfidf_{i}' for i in range(tfidf_features_df.shape[1])]

X = pd.concat([tfidf_features_df, numerical_features], axis=1)

X.columns = X.columns.astype(str)

y = data['is_trendy']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestClassifier(random_state=42, n_estimators=100)
rf_model.fit(X_train, y_train)

lr_model = LogisticRegression(max_iter=500, random_state=42)
lr_model.fit(X_train, y_train)

rf_predictions = rf_model.predict(X_test)
print("Random Forest Performance:")
print(classification_report(y_test, rf_predictions))
print(f"Accuracy: {accuracy_score(y_test, rf_predictions)}")

lr_predictions = lr_model.predict(X_test)
print("Logistic Regression Performance:")
print(classification_report(y_test, lr_predictions))
print(f"Accuracy: {accuracy_score(y_test, lr_predictions)}")

importances = rf_model.feature_importances_
feature_names = tfidf.get_feature_names_out().tolist() + ['num_comments', 'upvote_ratio', 'is_self']

feature_importances = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importances = feature_importances.sort_values(by='Importance', ascending=False).head(10)

print("Top 10 Features Influencing Trends:")
print(feature_importances)

feature_importances.to_csv('feature_importances.csv', index=False)
