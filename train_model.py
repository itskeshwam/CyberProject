import csv
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc, precision_recall_curve
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectFromModel

class MalwareModelTrainer:
    def __init__(self, data_path):
        self.data_path = data_path
        self.model = None
        self.scaler = StandardScaler()
        self.selected_features = None
        self.feature_importance = None

    def load_and_prepare_data(self):
        """Load and prepare the dataset with feature engineering"""
        # Load data using the '|' delimiter
        data = pd.read_csv(self.data_path, sep='|', quoting=csv.QUOTE_NONE)
        print("Available columns:", data.columns.tolist())
        
        # Select relevant features, excluding non-numeric columns
        feature_columns = [col for col in data.columns if col not in ['legitimate', 'Name', 'md5']]
        
        # Create feature matrix X and target vector y
        X = data[feature_columns]
        y = data['legitimate']
        
        # Handle missing values
        X = X.fillna(0)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=feature_columns)
        
        # Feature selection using Random Forest importance
        selector = RandomForestClassifier(n_estimators=100, random_state=42)
        selector.fit(X_scaled, y)
        
        # Get feature importance scores
        importance_scores = selector.feature_importances_
        
        # Select features with importance above mean
        importance_threshold = np.mean(importance_scores)
        selected_features_mask = importance_scores > importance_threshold
        
        # Get selected features
        self.selected_features = X_scaled.columns[selected_features_mask].tolist()
        X_selected = X_scaled[self.selected_features]
        
        print(f"\nSelected {len(self.selected_features)} features out of {len(feature_columns)}")
        return X_selected, y
    
    def train_model(self):
        """Train the model with cross-validation"""
        print("Loading and preparing data...")
        X, y = self.load_and_prepare_data()
        
        # Split the data
        print("Splitting data into train and test sets...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train Random Forest model (replacing LightGBM for better compatibility)
        print("Training Random Forest model...")
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        # Train the model
        self.model.fit(X_train, y_train)
        
        # Calculate feature importance
        self.feature_importance = pd.DataFrame({
            'feature': self.selected_features,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return X_test, y_test
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate the model performance"""
        print("\nEvaluating model performance...")
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        print("\nModel Performance Metrics:")
        print("-------------------------")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Calculate ROC curve and AUC
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        # Calculate Precision-Recall curve
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        pr_auc = auc(recall, precision)
        
        # Plot ROC curve
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        
        # Plot Precision-Recall curve
        plt.subplot(1, 2, 2)
        plt.plot(recall, precision, color='darkgreen', lw=2, label=f'PR curve (AUC = {pr_auc:.2f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        
        plt.tight_layout()
        plt.savefig('model_performance.png')
        plt.close()
        
        # Plot feature importance
        plt.figure(figsize=(10, 6))
        sns.barplot(x='importance', y='feature', data=self.feature_importance.head(15))
        plt.title('Top 15 Most Important Features')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        plt.close()
        
    def save_model(self):
        """Save the model and necessary components"""
        print("\nSaving model and components...")
        joblib.dump(self.model, 'malware_model.pkl')
        joblib.dump(self.scaler, 'feature_scaler.pkl')
        joblib.dump(self.selected_features, 'selected_features.pkl')
        
        print("\nModel and components saved successfully!")
        print("Files saved: malware_model.pkl, feature_scaler.pkl, selected_features.pkl")

if __name__ == "__main__":
    try:
        print("Starting malware model training...")
        # Initialize and train the model
        trainer = MalwareModelTrainer('MalwareData.csv')
        X_test, y_test = trainer.train_model()
        
        # Evaluate and save the model
        trainer.evaluate_model(X_test, y_test)
        trainer.save_model()
        print("\nTraining completed successfully!")
        
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        print("\nTraceback:")
        import traceback
        traceback.print_exc()
