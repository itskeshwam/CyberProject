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
        data = pd.read_csv(self.data_path, sep='|', quoting=csv.QUOTE_NONE)
        
        # Define specific features to use
        self.selected_features = [
            'Machine', 'SizeOfOptionalHeader', 'Characteristics', 
            'MajorLinkerVersion', 'MinorLinkerVersion', 'SizeOfCode',
            'SizeOfInitializedData', 'SizeOfUninitializedData', 
            'AddressOfEntryPoint', 'BaseOfCode', 'ImageBase',
            'SectionsMeanEntropy', 'SectionsMinEntropy'
        ]
        
        X = data[self.selected_features]
        y = data['legitimate']
        X = X.fillna(0)
        
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=self.selected_features)
        
        return X_scaled, y
    

    def train_model(self):
        """Train the model with cross-validation and detailed feature analysis"""
        print("\nLoading and preparing data...")
        X, y = self.load_and_prepare_data()
        
        # Split the data
        print("\nSplitting data into train and test sets...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train Random Forest model
        print("\nTraining Random Forest model...")
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        # Train the model
        self.model.fit(X_train, y_train)
        
        # Calculate and store feature importance with more detail
        self.feature_importance = pd.DataFrame({
            'feature': self.selected_features,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Print detailed feature importance analysis
        print("\nFeature Importance Analysis:")
        print("---------------------------")
        print(f"Total features selected: {len(self.selected_features)}")
        print("\nTop features by importance:")
        for idx, row in self.feature_importance.iterrows():
            print(f"{row['feature']:<30} {row['importance']:.4f}")
        
        # Calculate cumulative importance
        self.feature_importance['cumulative_importance'] = self.feature_importance['importance'].cumsum()
        print("\nCumulative importance analysis:")
        for idx, row in self.feature_importance.iterrows():
            print(f"Top {idx + 1} features explain {row['cumulative_importance']:.1%} of total importance")
            if row['cumulative_importance'] > 0.95:  # Stop after explaining 95% of importance
                break
        
        # Save detailed feature information
        feature_info = {
            'selected_features': self.selected_features,
            'importance_scores': self.feature_importance.to_dict(),
            'feature_stats': {
                'total_features': len(self.selected_features),
                'importance_threshold': np.mean(self.model.feature_importances_),
                'features_95_percent': len(self.feature_importance[self.feature_importance['cumulative_importance'] <= 0.95])
            }
        }
        joblib.dump(feature_info, 'feature_info.pkl')
        
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
