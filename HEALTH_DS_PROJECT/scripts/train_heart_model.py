import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    roc_curve, auc, precision_recall_curve
)
from scipy.stats import randint, uniform

def plot_feature_importance(model, feature_names, output_path="models/feature_importance.png"):
    """Plot feature importance."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(12, 8))
    plt.title("Feature Importances")
    plt.bar(range(len(importances)), importances[indices])
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_confusion_matrix(y_true, y_pred, output_path="models/confusion_matrix.png"):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_roc_curve(y_true, y_pred_proba, output_path="models/roc_curve.png"):
    """Plot ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def train_model():
    # Load prepared data
    X_train = joblib.load("data/processed/X_train.pkl")
    X_test = joblib.load("data/processed/X_test.pkl")
    y_train = joblib.load("data/processed/y_train.pkl")
    y_test = joblib.load("data/processed/y_test.pkl")

    print("Training set shape:", X_train.shape)
    print("Test set shape:", X_test.shape)

    # Define hyperparameter search space
    param_dist = {
        'n_estimators': randint(100, 500),
        'max_depth': [None] + list(range(10, 50, 5)),
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 10),
        'max_features': ['sqrt', 'log2', None],
        'class_weight': ['balanced', 'balanced_subsample'],
        'criterion': ['gini', 'entropy'],
        'bootstrap': [True, False]
    }

    # Initialize Random Forest
    rf = RandomForestClassifier(random_state=42)

    # Random search with cross-validation
    random_search = RandomizedSearchCV(
        rf, param_distributions=param_dist,
        n_iter=100, cv=5, scoring='roc_auc',
        n_jobs=-1, random_state=42, verbose=1
    )

    # Fit model
    print("\nPerforming hyperparameter search...")
    random_search.fit(X_train, y_train)

    # Get best model
    best_model = random_search.best_estimator_
    print("\nBest parameters:", random_search.best_params_)

    # Make predictions
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Generate and save plots
    plot_feature_importance(best_model, X_train.columns)
    plot_confusion_matrix(y_test, y_pred)
    plot_roc_curve(y_test, y_pred_proba)

    # Save feature importance scores
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    feature_importance.to_csv('models/feature_importance.csv', index=False)

    # Save model and important thresholds
    threshold_metrics = pd.DataFrame({
        'threshold': np.arange(0.1, 1.0, 0.1),
        'precision': [precision_score(y_test, y_pred_proba >= thresh) for thresh in np.arange(0.1, 1.0, 0.1)],
        'recall': [recall_score(y_test, y_pred_proba >= thresh) for thresh in np.arange(0.1, 1.0, 0.1)]
    })
    threshold_metrics.to_csv('models/threshold_metrics.csv', index=False)

    joblib.dump(best_model, "models/heart_model_improved.pkl")
    print("\nModel saved as 'heart_model_improved.pkl'")

    return best_model, feature_importance, threshold_metrics

if __name__ == "__main__":
    # Import these here to avoid circular imports
    from sklearn.metrics import precision_score, recall_score
    import os

    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)

    try:
        best_model, feature_importance, threshold_metrics = train_model()
        
        # Print top 10 most important features
        print("\nTop 10 Most Important Features:")
        print(feature_importance.head(10))
        
        # Print threshold analysis
        print("\nThreshold Analysis:")
        print(threshold_metrics)
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")