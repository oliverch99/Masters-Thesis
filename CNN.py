import numpy as np
import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Flatten, MaxPooling1D, Dropout
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
import shap
from keras.callbacks import EarlyStopping
from keras.regularizers import l2

# Set working directory
os.chdir("C://Users/Oliver/OneDrive - Aalborg Universitet/P10/kode/wsb-ml-trades-main - Kopi - Copy - Copy")

# Load data
with open("data/model_data.pkl", "rb") as file:
    data = pickle.load(file)


# Define features

features = ['score', 'sentiment', 'open', 'high',
            'low', 'close', 'volume', 'num_trades', 'vwap', 
            'log_return_close', 'title_sentiment', 
            'description_sentiment', 'title_length', 'description_length', 
            'title_pos_keywords', 'title_neg_keywords', 'hour', 'minute']

features = ['num_comments', 'score', 'sentiment', 'open', 
            'high', 'low', 'close', 'volume', 'num_trades',
            'vwap', 'log_return_close', 
            'title_sentiment', 'description_sentiment', 
            'title_length', 'description_length', 
            'title_pos_keywords', 'title_neg_keywords', 
            'description_pos_keywords', 'description_neg_keywords', 
            'hour', 'minute']


while len(features)>1:
    #features = ['vwap', 'log_return_close', 'num_trades', 'sentiment']
    # Split data into features and label 
    # seed = 3
    # tf.random.set_seed(seed)
    X = data[features].copy()
    y = data['oneHourWinner'].copy() 

    # Instantiate scaler and fit on features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Reshape data for CNN
    X_reshaped = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)

    # Split data into train and test
    X_train_reshaped, X_test_reshaped, y_train, y_test = train_test_split(X_reshaped, y, test_size=0.2, shuffle=False)


    # Define a function to build the CNN model
    # Define the model
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', padding='same', input_shape=(X_train_reshaped.shape[1], 1)),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        Flatten(),
        Dense(128, activation='relu', kernel_regularizer=l2(0.05)),
        Dropout(0.4),
        Dense(1, activation='sigmoid')
    ])


    # Try different optimizers
    optimizer = RMSprop(learning_rate=0.001)  # You can also try RMSprop(learning_rate=0.001) or SGD(learning_rate=0.002)

    # Compile the model
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    # Implement early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # # Compile model
    # model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    class_weights = {0: 1, 1: 1}
    # # Train model
    
    history = model.fit(X_train_reshaped, y_train, epochs=10, batch_size=32, 
                        validation_data=(X_test_reshaped, y_test), class_weight=class_weights)

    # Evaluate model
    train_loss, train_accuracy = model.evaluate(X_train_reshaped, y_train, verbose=0)
    test_loss, test_accuracy = model.evaluate(X_test_reshaped, y_test, verbose=0)

    # Make predictions with the model on the test set
    y_test_proba = model.predict(X_test_reshaped)
    y_test_preds = (y_test_proba > 0.5).astype(int)

    # Evaluate the model on the test set
    test_classification_rep = classification_report(y_test, y_test_preds)
    test_roc_auc = roc_auc_score(y_test, y_test_proba)

    print(f'Test Accuracy: {test_accuracy}')
    print(f'Test Classification Report:\n{test_classification_rep}')
    print(f'Test ROC AUC Score: {test_roc_auc}')
    # Check for features with zero importance

    # Reduce the number of background samples for SHAP
    background_sample_size = 100  # Select a smaller number of samples
    background = X_train_reshaped[np.random.choice(X_train_reshaped.shape[0], background_sample_size, replace=False)]

    # Calculate SHAP values using DeepExplainer
    explainer = shap.DeepExplainer(model, background)
    shap_values = explainer.shap_values(X_test_reshaped)

    # Summarize the feature importance
    #shap.summary_plot(shap_values, X_test_reshaped, feature_names=features)

    # Calculate mean absolute SHAP values for each feature
    shap_values_array = np.array(shap_values[0])
    shap_importance = np.mean(np.abs(shap_values_array), axis=1)[:,0]
    feature_importance = pd.DataFrame({'Feature': features, 'Importance': shap_importance})
    feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

    print("Feature importance based on SHAP values:")
    print(feature_importance)
    zero_importance_features = feature_importance[feature_importance['Importance'] == 0]['Feature'].tolist()
    
    # Remove features with zero importance, else remove the least important feature
    if zero_importance_features:
        for feature in zero_importance_features:
            features.remove(feature)
            print(f"Removing zero importance feature: {feature}")
    else:
        least_important_feature = feature_importance.iloc[-1]['Feature']
        features.remove(least_important_feature)
        print(f"Removing least important feature: {least_important_feature}")




with open("model.pkl", "wb") as f:
    pickle.dump((X_train_reshaped, X_test_reshaped, y_train, y_test, y_test_proba, y_test_preds, model),f)




import numpy as np
import pandas as pd
import os
import pickle
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Flatten, MaxPooling1D, Dropout
from tensorflow.keras.optimizers import Adam
import shap

# Set working directory
os.chdir("C://Users/Oliver/OneDrive - Aalborg Universitet/P10/kode/wsb-ml-trades-main - Kopi - Copy - Copy")

# Load data
with open("data/model_data.pkl", "rb") as file:
    data = pickle.load(file)

# Define features
features = ['hour', 'num_trades', 'title_neg_keywords', 'close', 'minute', 'volume']


#features = ['num_comments', 'score', 'sentiment', 'open', 'high',
            # 'low', 'close', 'volume', 'num_trades', 'vwap', 
            # 'log_return_close', 'title_sentiment', 
            # 'description_sentiment', 'title_length', 'description_length', 
            # 'title_pos_keywords', 'title_neg_keywords', 'description_pos_keywords', 
            # 'description_neg_keywords', 'hour', 'minute']

# Split data into features and label
X = data[features].copy()
y = data['oneHourWinner'].copy()

# Instantiate scaler and fit on features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reshape data for CNN
X_reshaped = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)

# Define a function to build the CNN model
def build_model(input_shape):
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', padding='same', input_shape=(X_train_reshaped.shape[1], 1)),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        Flatten(),
        Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(0.4),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Implement early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Perform cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=25)
cv_accuracies = []
cv_roc_aucs = []

for train_index, test_index in kf.split(X_reshaped):
    X_train_reshaped, X_test_reshaped = X_reshaped[train_index], X_reshaped[test_index]
    y_train, y_test = y[train_index], y[test_index]

    model = build_model((X_train_reshaped.shape[1], 1))

    # Using class weights
    class_weights = {0: 1, 1: 1.1}

    # Train the model with class weights
    model.fit(X_train_reshaped, y_train, epochs=10, batch_size=32, validation_data=(X_test_reshaped, y_test), class_weight=class_weights, verbose=0)

    # Evaluate model
    y_test_proba = model.predict(X_test_reshaped)
    y_test_preds = (y_test_proba > 0.5).astype(int)

    test_accuracy = accuracy_score(y_test, y_test_preds)
    test_roc_auc = roc_auc_score(y_test, y_test_proba)

    cv_accuracies.append(test_accuracy)
    cv_roc_aucs.append(test_roc_auc)

print(f'Cross-validated accuracies: {cv_accuracies}')
print(f'Cross-validated ROC AUC scores: {cv_roc_aucs}')
print(f'Mean CV accuracy: {np.mean(cv_accuracies)}')
print(f'Mean CV ROC AUC score: {np.mean(cv_roc_aucs)}')
