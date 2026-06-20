# bias_detection.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
import shap
import warnings
warnings.filterwarnings('ignore')

class BiasDetector:
    """A comprehensive bias detection and mitigation tool"""
    
    def __init__(self, data_path='adult.data'):
        self.data_path = data_path
        self.data = None
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.sensitive_attrs = ['race', 'sex']
        
    def load_data(self):
        """Load and clean the Adult dataset"""
        column_names = [
            'age', 'workclass', 'fnlwgt', 'education', 'education-num',
            'marital-status', 'occupation', 'relationship', 'race', 'sex',
            'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'
        ]
        
        try:
            self.data = pd.read_csv(self.data_path, header=None, names=column_names)
        except:
            self.data = pd.read_csv(self.data_path + '.txt', header=None, names=column_names)
        
        # Clean data
        self.data = self.data.replace(' ?', pd.NA)
        self.data = self.data.replace('?', pd.NA)
        self.data = self.data.dropna()
        
        # Convert income to binary
        self.data['income'] = self.data['income'].apply(lambda x: 1 if '>50K' in str(x) else 0)
        
        print(f"✅ Data loaded: {self.data.shape[0]} rows, {self.data.shape[1]} columns")
        return self.data
    
    def prepare_data(self):
        """Prepare data for modeling"""
        X = self.data.drop('income', axis=1)
        y = self.data['income']
        
        # Encode categorical variables
        self.categorical_cols = X.select_dtypes(include=['object']).columns
        self.label_encoders = {}
        
        for col in self.categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            self.label_encoders[col] = le
        
        # Save feature names for SHAP
        self.feature_names = X.columns.tolist()
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"✅ Data prepared: {self.X_train.shape[0]} train, {self.X_test.shape[0]} test")
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_model(self):
        """Train a Random Forest model"""
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(self.X_train, self.y_train)
        
        # Evaluate
        y_pred = self.model.predict(self.X_test)
        accuracy = (y_pred == self.y_test).mean()
        
        print(f"✅ Model trained with accuracy: {accuracy*100:.2f}%")
        return self.model, accuracy
    
    def calculate_bias_metrics(self):
        """Calculate fairness metrics for sensitive attributes"""
        bias_report = {}
        
        for attr in self.sensitive_attrs:
            if attr in self.feature_names:
                # Get predictions
                y_pred = self.model.predict(self.X_test)
                
                # Get original values for this attribute
                attr_idx = self.feature_names.index(attr)
                attr_values = self.X_test.iloc[:, attr_idx]
                
                # Convert back to original labels
                if attr in self.label_encoders:
                    original_values = self.label_encoders[attr].inverse_transform(attr_values)
                else:
                    original_values = attr_values
                
                # Calculate positive prediction rate per group
                df_results = pd.DataFrame({
                    'predicted': y_pred,
                    'actual': self.y_test.values,
                    'group': original_values
                })
                
                group_metrics = df_results.groupby('group').agg({
                    'predicted': ['mean', 'count'],
                    'actual': 'mean'
                }).round(3)
                
                group_metrics.columns = ['predicted_positive_rate', 'count', 'actual_positive_rate']
                
                bias_report[attr] = group_metrics
                
                # Calculate disparity
                max_rate = group_metrics['predicted_positive_rate'].max()
                min_rate = group_metrics['predicted_positive_rate'].min()
                disparity = max_rate / min_rate if min_rate > 0 else float('inf')
                
                bias_report[f'{attr}_disparity'] = disparity
                bias_report[f'{attr}_bias_level'] = 'High' if disparity > 1.5 else 'Medium' if disparity > 1.2 else 'Low'
        
        return bias_report
    
    def explain_predictions(self, num_samples=100):
        """Explain model predictions using SHAP"""
        print("🔍 Generating SHAP explanations...")
        
        # Create SHAP explainer
        explainer = shap.TreeExplainer(self.model)
        
        # Get SHAP values for test samples
        shap_values = explainer.shap_values(self.X_test[:num_samples])
        
        # For binary classification, use the positive class (index 1)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        # Create summary
        shap_summary = {
            'shap_values': shap_values,
            'feature_names': self.feature_names,
            'X_test_sample': self.X_test[:num_samples]
        }
        
        return shap_summary, explainer
    
    def generate_counterfactual(self, sample_idx=0):
        """Generate a counterfactual explanation for a specific prediction"""
        # Get the sample
        sample = self.X_test.iloc[[sample_idx]]
        original_pred = self.model.predict(sample)[0]
        
        # Try to find a counterfactual by perturbing features
        counterfactuals = []
        
        # For each feature, try to change it and see if prediction changes
        for i, feature in enumerate(self.feature_names):
            modified_sample = sample.copy()
            
            # Get feature value range
            feature_values = self.X_train.iloc[:, i]
            
            # Try different values
            for value in np.percentile(feature_values, [25, 50, 75]):
                modified_sample.iloc[0, i] = value
                new_pred = self.model.predict(modified_sample)[0]
                
                if new_pred != original_pred:
                    counterfactuals.append({
                        'feature': feature,
                        'original_value': sample.iloc[0, i],
                        'new_value': value,
                        'new_prediction': '>50K' if new_pred == 1 else '<=50K'
                    })
                    break
        
        return {
            'original_sample': sample,
            'original_prediction': '>50K' if original_pred == 1 else '<=50K',
            'counterfactuals': counterfactuals[:5]  # Return top 5
        }

# Main execution
if __name__ == "__main__":
    # Create detector
    detector = BiasDetector()
    
    # Load and prepare data
    detector.load_data()
    detector.prepare_data()
    
    # Train model
    detector.train_model()
    
    # Calculate bias metrics
    bias_report = detector.calculate_bias_metrics()
    
    print("\n" + "="*50)
    print("📊 BIAS DETECTION REPORT")
    print("="*50)
    
    for attr in ['race', 'sex']:
        if attr in bias_report:
            print(f"\n🔹 {attr.upper()}:")
            print(bias_report[attr])
            print(f"   Disparity Ratio: {bias_report[f'{attr}_disparity']:.2f}x")
            print(f"   Bias Level: {bias_report[f'{attr}_bias_level']}")
    
    # Generate explanations
    shap_summary, explainer = detector.explain_predictions(50)
    
    print("\n✅ Bias detection complete!")