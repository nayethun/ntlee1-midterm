# %%
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, HistGradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from scipy.sparse import hstack, csr_matrix
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from textblob import TextBlob

# %%
# Load data
train_df = pd.read_csv('/Users/ReconGuest/Desktop/cs506/train.csv')
test_df = pd.read_csv('/Users/ReconGuest/Desktop/cs506/test.csv')

# Split labeled and unlabeled data
train_labeled = train_df[train_df['Score'].notnull()].copy()
train_unlabeled = train_df[train_df['Score'].isnull()].copy()

# Sample 10% of labeled training data
train_labeled = train_labeled.sample(frac=0.1, random_state=42).reset_index(drop=True)

# Extract test data from train_unlabeled using matching IDs from test_df
test_data = train_unlabeled[train_unlabeled['Id'].isin(test_df['Id'])].copy()


# %%
# Add missing rows to match all IDs in test_df
missing_ids = set(test_df['Id']) - set(test_data['Id'])
missing_rows = pd.DataFrame({'Id': list(missing_ids), 'Text': [''] * len(missing_ids)})
test_data = pd.concat([test_data, missing_rows], ignore_index=True)
test_data = test_data.set_index('Id').reindex(test_df['Id']).reset_index()


# %%
# Simple text preprocessing function
def preprocess_text(text):
    if not isinstance(text, str) or not text.strip():
        return ''
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    words = text.split()  # Split into words
    stopwords = set(['the', 'a', 'an', 'and', 'or', 'in', 'on', 'at', 'to', 'for', 'with', 'of'])
    words = [word for word in words if word not in stopwords]
    return ' '.join(words)

# Apply preprocessing with a progress bar
tqdm.pandas(desc="Processing Training Data")
train_labeled['processed_text'] = train_labeled['Text'].fillna('').progress_apply(preprocess_text)

tqdm.pandas(desc="Processing Test Data")
test_data['processed_text'] = test_data['Text'].fillna('').progress_apply(preprocess_text)

# Add text length features
train_labeled['text_length'] = train_labeled['processed_text'].apply(lambda x: len(x.split()))
test_data['text_length'] = test_data['processed_text'].apply(lambda x: len(x.split()))

# Feature: Helpfulness Ratio
train_labeled['helpfulness_ratio'] = train_labeled['HelpfulnessNumerator'] / (train_labeled['HelpfulnessDenominator'] + 1)
test_data['helpfulness_ratio'] = test_data['HelpfulnessNumerator'] / (test_data['HelpfulnessDenominator'] + 1)
train_labeled['helpfulness_ratio'].fillna(0, inplace=True)
test_data['helpfulness_ratio'].fillna(0, inplace=True)

# Add sentiment feature
train_labeled['sentiment'] = train_labeled['processed_text'].apply(lambda x: TextBlob(x).sentiment.polarity)
test_data['sentiment'] = test_data['processed_text'].apply(lambda x: TextBlob(x).sentiment.polarity)

# Scaling numerical features
scaler = StandardScaler()
train_num_features = scaler.fit_transform(train_labeled[['helpfulness_ratio', 'text_length', 'sentiment']])
test_num_features = scaler.transform(test_data[['helpfulness_ratio', 'text_length', 'sentiment']])

# TF-IDF Vectorization
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_tfidf = tfidf.fit_transform(train_labeled['processed_text'])
X_test_tfidf = tfidf.transform(test_data['processed_text'])

# Dimensionality reduction with TruncatedSVD
svd = TruncatedSVD(n_components=100, random_state=42)
X_train_svd = svd.fit_transform(X_train_tfidf)
X_test_svd = svd.transform(X_test_tfidf)

# %%
# Combine all features
train_features = hstack([csr_matrix(train_num_features), csr_matrix(X_train_svd)])
test_features = hstack([csr_matrix(test_num_features), csr_matrix(X_test_svd)])

# Prepare target variable: Shift labels to start from 0
y_train = train_labeled['Score'].astype(int) - 1

# Define individual models
xgb_model = XGBClassifier(objective='multi:softmax', num_class=5, random_state=42)
lgbm_model = LGBMClassifier(objective='multiclass', num_class=5, random_state=42)
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight='balanced')
hgb_model = HistGradientBoostingClassifier(max_iter=100, max_depth=10, random_state=42)

# Model Calibration
calibrated_rf = CalibratedClassifierCV(rf_model, method='sigmoid', cv=3)
calibrated_hgb = CalibratedClassifierCV(hgb_model, method='sigmoid', cv=3)

# Voting Classifier
voting_clf = VotingClassifier(
    estimators=[
        ('xgb', xgb_model),
        ('lgbm', lgbm_model),
        ('rf', calibrated_rf),
        ('hgb', calibrated_hgb)
    ],
    voting='soft'
)



# %%
# Cross-Validation with StratifiedKFold
n_splits = 5
stratified_kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Cross-validation scores with single-threaded processing
cv_scores = cross_val_score(
    voting_clf, 
    train_features.toarray(), 
    y_train, 
    cv=stratified_kfold, 
    scoring='accuracy', 
    n_jobs=1  # Set n_jobs to 1
)

# Display cross-validation results
print(f"Cross-Validation Accuracy for {n_splits} folds: {cv_scores}")
print(f"Mean Cross-Validation Accuracy: {cv_scores.mean():.4f}")

# %%
# Fit the model on the entire training set
voting_clf.fit(train_features.toarray(), y_train)

# Predict on the test set and shift back to original labels
test_predictions = voting_clf.predict(test_features.toarray()) + 1

# Prepare submission file
submission = pd.DataFrame({'Id': test_df['Id'], 'Score': test_predictions})
submission.to_csv('submission.csv', index=False)

print("Submission file created with shape:", submission.shape)

# %%
# Predict on the validation set
y_valid_pred = voting_clf.predict(train_features.toarray())

# Convert back to original labels (1 to 5)
y_valid_original = y_train + 1
y_valid_pred_original = y_valid_pred + 1

# Confusion matrix
cm = confusion_matrix(y_valid_original, y_valid_pred_original, labels=[1, 2, 3, 4, 5])

# Plot the confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(1, 6), yticklabels=range(1, 6))
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()


