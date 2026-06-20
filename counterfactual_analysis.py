# counterfactual_analysis.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class CounterfactualAnalyzer:
    """Advanced counterfactual analysis for bias detection"""
    
    def __init__(self, data_path='adult.data'):
        self.data_path = data_path
        self.data = None
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        
    def load_data(self):
        """Load the Adult dataset"""
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
        
        print(f"✅ Data loaded: {self.data.shape[0]} rows")
        return self.data
    
    def prepare_data(self):
        """Prepare data for counterfactual analysis"""
        X = self.data.drop('income', axis=1)
        y = self.data['income']
        
        # Save categorical columns
        self.categorical_cols = X.select_dtypes(include=['object']).columns
        self.label_encoders = {}
        
        # Encode categorical
        for col in self.categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            self.label_encoders[col] = le
        
        self.feature_names = X.columns.tolist()
        self.protected_attrs = ['race', 'sex']
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"✅ Data prepared: {self.X_train.shape[0]} train, {self.X_test.shape[0]} test")
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_model(self):
        """Train the model"""
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(self.X_train, self.y_train)
        
        accuracy = self.model.score(self.X_test, self.y_test)
        print(f"✅ Model accuracy: {accuracy*100:.2f}%")
        return self.model
    
    def find_counterfactuals(self, sample_idx=0, max_attempts=100):
        """Find counterfactual explanations for a specific instance"""
        sample = self.X_test.iloc[[sample_idx]].copy()
        original_pred = self.model.predict(sample)[0]
        
        print(f"\n🔍 Finding counterfactuals for sample {sample_idx}")
        print(f"   Original prediction: {'>50K' if original_pred == 1 else '<=50K'}")
        
        counterfactuals = []
        
        # Try to find counterfactuals by modifying features
        for feature_idx, feature_name in enumerate(self.feature_names):
            if feature_name in self.protected_attrs:
                continue  # Don't modify protected attributes
                
            # Get feature statistics
            feature_values = self.X_train.iloc[:, feature_idx]
            mean_val = feature_values.mean()
            std_val = feature_values.std()
            
            # Try different modifications
            modifications = [
                mean_val,
                mean_val + std_val,
                mean_val - std_val,
                feature_values.min(),
                feature_values.max()
            ]
            
            for new_val in modifications:
                modified_sample = sample.copy()
                modified_sample.iloc[0, feature_idx] = new_val
                
                new_pred = self.model.predict(modified_sample)[0]
                
                if new_pred != original_pred:
                    counterfactuals.append({
                        'feature': feature_name,
                        'original_value': sample.iloc[0, feature_idx],
                        'modified_value': new_val,
                        'change': new_val - sample.iloc[0, feature_idx],
                        'new_prediction': '>50K' if new_pred == 1 else '<=50K',
                        'success': True
                    })
                    break  # Found one counterfactual for this feature
        
        return {
            'sample_idx': sample_idx,
            'original_prediction': '>50K' if original_pred == 1 else '<=50K',
            'counterfactuals': counterfactuals,
            'num_found': len(counterfactuals)
        }
    
    def analyze_group_bias(self):
        """Analyze bias across different demographic groups using counterfactuals"""
        print("\n⚖️ Counterfactual Bias Analysis")
        print("="*60)
        
        # Analyze each protected attribute
        bias_results = {}
        
        for attr in self.protected_attrs:
            if attr in self.feature_names:
                attr_idx = self.feature_names.index(attr)
                attr_values = self.X_test.iloc[:, attr_idx]
                
                # Get unique groups
                unique_groups = np.unique(attr_values)
                group_names = [self.label_encoders[attr].inverse_transform([g])[0] for g in unique_groups]
                
                print(f"\n🔹 Analyzing bias by '{attr}':")
                
                group_stats = {}
                for group, group_name in zip(unique_groups, group_names):
                    # Get indices for this group
                    group_indices = np.where(attr_values == group)[0]
                    
                    if len(group_indices) > 0:
                        # Get predictions for this group
                        group_predictions = self.model.predict(self.X_test.iloc[group_indices])
                        positive_rate = group_predictions.mean()
                        
                        group_stats[group_name] = {
                            'count': len(group_indices),
                            'positive_rate': positive_rate,
                            'percentage': len(group_indices) / len(self.X_test) * 100
                        }
                
                # Display results
                for group_name, stats in group_stats.items():
                    print(f"   {group_name}: {stats['positive_rate']*100:.1f}% positive ({stats['count']} samples)")
                
                # Calculate disparity
                rates = [stats['positive_rate'] for stats in group_stats.values()]
                if len(rates) > 1:
                    disparity = max(rates) / min(rates) if min(rates) > 0 else float('inf')
                    print(f"   📊 Disparity Ratio: {disparity:.2f}x")
                    
                    if disparity > 1.5:
                        print(f"   ⚠️  HIGH BIAS detected!")
                    elif disparity > 1.2:
                        print(f"   ⚡ Moderate bias detected")
                    else:
                        print(f"   ✅ Low bias detected")
                
                bias_results[attr] = group_stats
        
        return bias_results
    
    def generate_recommendations(self):
        """Generate recommendations to mitigate bias"""
        recommendations = []
        
        # Analyze model behavior
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        recommendations.append({
            'type': 'Feature Importance',
            'message': 'Top features driving predictions:',
            'details': feature_importance.head(5).to_dict('records')
        })
        
        # Check for protected attribute importance
        for attr in self.protected_attrs:
            if attr in feature_importance['feature'].values:
                importance = feature_importance[feature_importance['feature'] == attr]['importance'].values[0]
                if importance > 0.01:
                    recommendations.append({
                        'type': 'Bias Warning',
                        'message': f"'{attr}' has high importance ({importance:.3f}) in the model",
                        'details': f"Consider removing or de-biasing '{attr}' from the model"
                    })
        
        return recommendations

# Main execution
if __name__ == "__main__":
    # Create analyzer
    analyzer = CounterfactualAnalyzer()
    
    # Load and prepare data
    analyzer.load_data()
    analyzer.prepare_data()
    
    # Train model
    analyzer.train_model()
    
    # Analyze bias
    bias_results = analyzer.analyze_group_bias()
    
    # Find counterfactuals for a specific sample
    print("\n" + "="*60)
    print("🔍 COUNTERFACTUAL EXAMPLES")
    print("="*60)
    
    # Find counterfactuals for a minority sample
    results = analyzer.find_counterfactuals(sample_idx=0)
    print(f"\n🔹 Results for sample {results['sample_idx']}:")
    print(f"   Original Prediction: {results['original_prediction']}")
    
    if results['counterfactuals']:
        print(f"\n   Found {results['num_found']} counterfactuals:")
        for cf in results['counterfactuals'][:3]:
            print(f"   - Change '{cf['feature']}' from {cf['original_value']:.1f} to {cf['modified_value']:.1f}")
            print(f"     → Prediction would be: {cf['new_prediction']}")
    else:
        print("   No counterfactuals found")
    
    # Generate recommendations
    recommendations = analyzer.generate_recommendations()
    print("\n" + "="*60)
    print("💡 RECOMMENDATIONS")
    print("="*60)
    
    for rec in recommendations:
        print(f"\n🔹 {rec['type']}:")
        print(f"   {rec['message']}")
        if 'details' in rec:
            print(f"   Details: {rec['details']}")