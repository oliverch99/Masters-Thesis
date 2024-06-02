import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

def evaluate_classifier(clf, X, y, max_iter, test_size):
    clf_name = clf.__class__.__name__
    if hasattr(clf, 'max_iter'):
        clf.max_iter = max_iter

    print(f"Evaluating classifier: {clf_name}")

    # Data Preprocessing
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Initialize RFE
    rfe = RFE(clf, n_features_to_select=1)
    rfe.fit(X_scaled, y)

    feature_ranking = list(rfe.ranking_)
    selected_features = list(X.columns)
    ranked_features = sorted(zip(selected_features, feature_ranking), key=lambda item: item[1])
    
    print(f"Initial feature ranking completed for {clf_name}")

    results = []
    feature_removal_order = []
    
    while len(selected_features) > 1:
        print(f"Evaluating with {len(selected_features)} features for {clf_name}")

        # Select features based on the current ranking
        X_rfe = X[selected_features]
        X_scaled_rfe = scaler.fit_transform(X_rfe)

        # Train-Test Split
        X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(X_scaled_rfe, y, test_size=test_size, random_state=25)

        try:
            # Cross-Validation
            scores = cross_val_score(clf, X_scaled_rfe, y, cv=10)
            
            # Train the final model
            clf.fit(X_train_scaled, y_train)

            # Make Predictions
            preds = clf.predict(X_test_scaled)

            # Model Evaluation
            report = classification_report(y_test, preds, output_dict=True)

            results.append({
                'Classifier': clf_name,
                'Number of Features': len(selected_features),
                'Selected Features': selected_features.copy(),
                'Cross-Validation Accuracy': f"{scores.mean():.2f} ± {scores.std():.2f}",
                'Classification Report': report
            })
            
            print(f"Completed evaluation with {len(selected_features)} features for {clf_name}")

            # Remove the least important feature
            least_important_feature = ranked_features.pop(0)[0]
            feature_removal_order.append(least_important_feature)
            selected_features.remove(least_important_feature)

            if len(selected_features) > 1:
                # Recompute RFE with the remaining features
                X_rfe = X[selected_features]
                X_scaled_rfe = scaler.fit_transform(X_rfe)
                rfe = RFE(clf, n_features_to_select=1)
                rfe.fit(X_scaled_rfe, y)
                feature_ranking = list(rfe.ranking_)
                ranked_features = sorted(zip(selected_features, feature_ranking), key=lambda item: item[1])

        except Exception as e:
            print(f"Error during model training or evaluation with {clf_name} and {len(selected_features)} features: {e}")
            break

    return results, {clf_name: pd.DataFrame(feature_removal_order, columns=['Removed Features'])}

def evaluate_classifiers(data, target_column, classifiers, max_iter=500, test_size=0.2):
    results = []
    rankings = {}
    
    # Split features and target
    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    for clf in classifiers:
        res, rank = evaluate_classifier(clf, X, y, max_iter, test_size)
        results.extend(res)
        rankings.update(rank)
        print(f"Completed evaluation for classifier: {clf.__class__.__name__}")
    
    return results, rankings

# Load data
with open("data/model_data.pkl", "rb") as file:
    data = pickle.load(file)

# Define classifiers
classifiers = [
    xgb.XGBClassifier(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    GradientBoostingClassifier(),
    LogisticRegression()
]

# Example usage
target_column = 'oneHourWinner'  # Replace with actual target column name

# Filter numeric features
features = [feature for feature in data.columns if pd.api.types.is_numeric_dtype(data[feature]) & (feature not in ['tenMinuteWinner', 'thirtyMinuteWinner'])]

results, rankings = evaluate_classifiers(data[features], target_column, classifiers)
