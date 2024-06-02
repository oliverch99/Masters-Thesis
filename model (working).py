import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.feature_selection import RFE
import pickle
import os

# Set working directory
os.chdir("C://Users/Oliver/OneDrive - Aalborg Universitet/P10/kode/wsb-ml-trades-main - Kopi - Copy - Copy")

# Load data
with open("data/model_data.pkl", "rb") as file:
    data = pickle.load(file)

# Define features
features = [ 'num_comments', 'score', 'close', 'volume', 
            'num_trades', 'log_return_close', 'title_sentiment', 
            'title_length', 'title_pos_keywords', 'title_neg_keywords', 
            'description_sentiment', 'description_length', 
            'description_pos_keywords', 'description_neg_keywords', 
            'hour', 'minute']

# Split data into features and label 
X = data[features].copy()
y = data['oneHourWinner'].copy() 

# Instantiate scaler and fit on features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into train and test
X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=25)

# Instantiate the model
tree = DecisionTreeClassifier()

# Perform RFE
rfe = RFE(estimator=tree, n_features_to_select=1, step=1)
rfe.fit(X_train_scaled, y_train)

# Get the ranking of features
ranking = rfe.ranking_

# Create a DataFrame to view feature rankings
feature_ranking = pd.DataFrame({'Feature': features, 'Rank': ranking})
feature_ranking = feature_ranking.sort_values(by='Rank')

print("Feature ranking:")
print(feature_ranking)

# Select the top N features (e.g., top 5)

N = 16
top_features = feature_ranking[feature_ranking['Rank'] <= N]['Feature'].tolist()
print(f"Top features: {top_features}")

# Retrain the model using only the top features
X_top_train = X_train_scaled[:, feature_ranking['Rank'] <= N]
X_top_test = X_test_scaled[:, feature_ranking['Rank'] <= N]

tree.fit(X_top_train, y_train)

# Make predictions with the model
tree_preds = tree.predict(X_top_test)
tree_proba = tree.predict_proba(X_top_test)

# Evaluate the model
accuracy = accuracy_score(y_test, tree_preds)
classification_rep = classification_report(y_test, tree_preds)
roc_auc = roc_auc_score(y_test, tree_proba[:, 1])

print(f'\n Accuracy: {accuracy}\n \
      \nClassification report: \n {classification_rep}\n \
      ROC AUC Score: {roc_auc}')
























import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.feature_selection import RFE
import pickle
import os

# Set working directory
os.chdir("C://Users/Oliver/OneDrive - Aalborg Universitet/P10/kode/wsb-ml-trades-main - Kopi - Copy - Copy")

# Load data
with open("data/model_data.pkl", "rb") as file:
    data = pickle.load(file)

# Define features
features = ['title_processed', 'description_processed', 
            'num_comments', 'score', 'close', 'volume', 
            'num_trades', 'log_return_close', 'title_sentiment', 
            'title_length', 'title_pos_keywords', 'title_neg_keywords', 
            'description_sentiment', 'description_length', 
            'description_pos_keywords', 'description_neg_keywords', 
            'hour', 'minute',
            'link_flair_text', 'market_segment']

# Split data into features and label 
X = data[features].copy()
y = data['oneHourWinner'].copy() 

# Instantiate scaler and fit on features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into train and test
X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=25)

# Instantiate the model
tree = DecisionTreeClassifier()

# Perform RFE
rfe = RFE(estimator=tree, n_features_to_select=5, step=1)  # Select top 5 features
rfe.fit(X_train_scaled, y_train)

# Get the ranking of features
ranking = rfe.ranking_

# Create a DataFrame to view feature rankings
feature_ranking = pd.DataFrame({'Feature': features, 'Rank': ranking})
feature_ranking = feature_ranking.sort_values(by='Rank')

print("Feature ranking:")
print(feature_ranking)

N = 16
# Select the top N features (e.g., top 5)
top_features = feature_ranking[feature_ranking['Rank'] <= N]['Feature'].tolist()
print(f"Top features: {top_features}")

# Retrain the model using only the top features
X_top_train = X_train_scaled[:, feature_ranking['Rank'] <= N]
X_top_test = X_test_scaled[:, feature_ranking['Rank'] <= N]

tree.fit(X_top_train, y_train)

# Make predictions with the model on the training set
tree_train_preds = tree.predict(X_top_train)
tree_train_proba = tree.predict_proba(X_top_train)

# Evaluate the model on the training set
train_accuracy = accuracy_score(y_train, tree_train_preds)
train_classification_rep = classification_report(y_train, tree_train_preds)
train_roc_auc = roc_auc_score(y_train, tree_train_proba[:, 1])

print(f'Training Accuracy: {train_accuracy}')
print(f'Training Classification Report:\n{train_classification_rep}')
print(f'Training ROC AUC Score: {train_roc_auc}')

# Make predictions with the model on the test set
tree_test_preds = tree.predict(X_top_test)
tree_test_proba = tree.predict_proba(X_top_test)

# Evaluate the model on the test set
test_accuracy = accuracy_score(y_test, tree_test_preds)
test_classification_rep = classification_report(y_test, tree_test_preds)
test_roc_auc = roc_auc_score(y_test, tree_test_proba[:, 1])

print(f'Test Accuracy: {test_accuracy}')
print(f'Test Classification Report:\n{test_classification_rep}')
print(f'Test ROC AUC Score: {test_roc_auc}')







import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.feature_selection import RFE
import pickle
import os
import numpy as np

# Set working directory
os.chdir("C://Users/Oliver/OneDrive - Aalborg Universitet/P10/kode/wsb-ml-trades-main - Kopi - Copy - Copy")

# Load data
with open("data/model_data.pkl", "rb") as file:
    data = pickle.load(file)
data
# Define features
features = ['title_processed', 'description_processed', 
            'num_comments', 'score', 'close', 'volume', 
            'num_trades', 'log_return_close', 'title_sentiment', 
            'title_length', 'title_pos_keywords', 'title_neg_keywords', 
            'description_sentiment', 'description_length', 
            'description_pos_keywords', 'description_neg_keywords', 
            'hour', 'minute',
            'link_flair_text', 'market_segment']

features = ['num_comments', 'score', 'close', 'volume', 
            'num_trades', 'log_return_close', 'title_sentiment', 
            'title_length', 'title_pos_keywords', 'title_neg_keywords', 
            'description_sentiment', 'description_length', 
            'description_pos_keywords', 'description_neg_keywords', 
            'hour', 'minute']

features = ['num_comments', 'score', 'sentiment', 'open', 'high',
            'low', 'close', 'volume', 'num_trades', 'vwap', 
            'log_return_close', 'oneHourWinner', 'title_sentiment', 
            'description_sentiment', 'title_length', 'description_length', 
            'title_pos_keywords', 'title_neg_keywords', 'description_pos_keywords', 
            'description_neg_keywords', 'hour', 'minute']

# Split data into features and label 
X = data[features].copy()
y = data['oneHourWinner'].copy() 

# Instantiate scaler and fit on features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into train and test
X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=25)

# Instantiate the model
tree = DecisionTreeClassifier()

# Perform RFE
rfe = RFE(estimator=tree, n_features_to_select=5, step=1)  # Select top 5 features
rfe.fit(X_train_scaled, y_train)

# Get the ranking of features
ranking = rfe.ranking_

# Create a DataFrame to view feature rankings
feature_ranking = pd.DataFrame({'Feature': features, 'Rank': ranking})
feature_ranking = feature_ranking.sort_values(by='Rank')

print("Feature ranking:")
print(feature_ranking)

N = 1
# Select the top N features (e.g., top 5)
top_features = feature_ranking[feature_ranking['Rank'] <= N]['Feature'].tolist()
print(f"Top features: {top_features}")

# Retrain the model using only the top features
X_top_train = X_train_scaled[:, (feature_ranking['Rank'] <= N).sort_index()]
X_top_test = X_test_scaled[:, (feature_ranking['Rank'] <= N).sort_index()]
X_top_train = X_train_scaled[:, 2:4]
X_top_test = X_test_scaled[:, 2:4]
# Implementing cross-validation
cross_val_scores = cross_val_score(tree, X.iloc[:, 2:4],y, cv=10, scoring='accuracy')
print(f'Cross-validation scores: {cross_val_scores}')
print(f'Mean cross-validation score: {np.mean(cross_val_scores)}')
print(f'Standard deviation of cross-validation scores: {np.std(cross_val_scores)}')

# Train the model on the entire training set
tree.fit(X_top_train, y_train)

# Make predictions with the model on the training set
tree_train_preds = tree.predict(X_top_train)
tree_train_proba = tree.predict_proba(X_top_train)

# Evaluate the model on the training set
train_accuracy = accuracy_score(y_train, tree_train_preds)
train_classification_rep = classification_report(y_train, tree_train_preds)
train_roc_auc = roc_auc_score(y_train, tree_train_proba[:, 1])

print(f'Training Accuracy: {train_accuracy}')
print(f'Training Classification Report:\n{train_classification_rep}')
print(f'Training ROC AUC Score: {train_roc_auc}')

# Make predictions with the model on the test set
tree_test_preds = tree.predict(X_top_test)
tree_test_proba = tree.predict_proba(X_top_test)

# Evaluate the model on the test set
test_accuracy = accuracy_score(y_test, tree_test_preds)
test_classification_rep = classification_report(y_test, tree_test_preds)
test_roc_auc = roc_auc_score(y_test, tree_test_proba[:, 1])

print(f'Test Accuracy: {test_accuracy}')
print(f'Test Classification Report:\n{test_classification_rep}')
print(f'Test ROC AUC Score: {test_roc_auc}')



