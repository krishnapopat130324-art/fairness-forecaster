import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import time
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Fairness Forecaster - 90%+ Accuracy",
    page_icon="⚖️",
    layout="wide"
)

st.title("⚖️ Fairness Forecaster - Bias Detection & Mitigation Tool")
st.markdown("---")

st.sidebar.header("📊 Controls")

@st.cache_data
def load_data():
    column_names = [
        'age', 'workclass', 'fnlwgt', 'education', 'education-num', 
        'marital-status', 'occupation', 'relationship', 'race', 'sex',
        'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'
    ]
    
    try:
        df = pd.read_csv('adult.data.txt', header=None, names=column_names)
    except:
        try:
            df = pd.read_csv('adult.data', header=None, names=column_names)
        except:
            from ucimlrepo import fetch_ucirepo
            adult = fetch_ucirepo(id=2)
            df = pd.concat([adult.data.features, adult.data.targets], axis=1)
            df.columns = column_names
    
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].str.strip()
    
    df = df.replace(' ?', pd.NA)
    df = df.replace('?', pd.NA)
    df = df.replace('', pd.NA)
    df = df.dropna()
    
    df['income'] = df['income'].apply(lambda x: 1 if '>50K' in str(x) else 0)
    
    return df

data = load_data()
st.sidebar.success(f"✅ Dataset loaded: {data.shape[0]} rows, {data.shape[1]} columns")

st.subheader("📋 Dataset Overview")
col1, col2, col3 = st.columns(3)
col1.metric("Total Rows", data.shape[0])
col2.metric("Total Columns", data.shape[1])
col3.metric("Income >50K", f"{data['income'].sum()} ({data['income'].mean()*100:.1f}%)")

with st.expander("🔍 View Data Sample"):
    st.dataframe(data.head(10))

st.subheader("📊 Exploratory Data Analysis")

feature = st.selectbox("Select a feature to analyze", data.columns[:-1])

tab1, tab2, tab3 = st.tabs(["Feature Distribution", "Income by Feature", "Correlation Matrix"])

with tab1:
    fig, ax = plt.subplots(figsize=(10, 5))
    
    if data[feature].dtype == 'object':
        data[feature].value_counts().head(10).plot(kind='bar', ax=ax)
        ax.set_title(f'Distribution of {feature}')
        ax.tick_params(axis='x', rotation=45)
    else:
        data[feature].hist(bins=30, ax=ax)
        ax.set_title(f'Distribution of {feature}')
    
    st.pyplot(fig)

with tab2:
    fig, ax = plt.subplots(figsize=(10, 5))
    
    if data[feature].dtype == 'object':
        data.groupby(feature)['income'].mean().sort_values().head(10).plot(kind='bar', ax=ax)
        ax.set_title(f'Income Rate by {feature}')
        ax.tick_params(axis='x', rotation=45)
    else:
        if data[feature].nunique() > 10:
            data.groupby(pd.cut(data[feature], bins=10))['income'].mean().plot(kind='line', ax=ax, marker='o')
        else:
            data.groupby(feature)['income'].mean().plot(kind='bar', ax=ax)
        ax.set_title(f'Income Rate by {feature}')
    
    st.pyplot(fig)

with tab3:
    numeric_data = data.copy()
    
    for col in numeric_data.columns:
        if numeric_data[col].dtype == 'object':
            le = LabelEncoder()
            numeric_data[col] = numeric_data[col].fillna('Unknown')
            numeric_data[col] = le.fit_transform(numeric_data[col].astype(str))
        elif numeric_data[col].dtype == 'int64' or numeric_data[col].dtype == 'float64':
            pass
        else:
            numeric_data[col] = numeric_data[col].astype(str)
            le = LabelEncoder()
            numeric_data[col] = le.fit_transform(numeric_data[col])
    
    fig, ax = plt.subplots(figsize=(12, 8))
    corr = numeric_data.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
    plt.title('Correlation Matrix')
    st.pyplot(fig)

st.subheader("🤖 Model Training & Bias Analysis")

def prepare_data(df):
    X = df.drop('income', axis=1).copy()
    y = df['income'].copy()
    
    for col in X.columns:
        if X[col].dtype == 'object':
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
        elif X[col].dtype == 'int64' or X[col].dtype == 'float64':
            pass
        else:
            X[col] = X[col].astype(str)
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
    
    for col in X.columns:
        if not np.issubdtype(X[col].dtype, np.number):
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    
    return X, y

X, y = prepare_data(data)

# Feature Engineering for better accuracy
# Create interaction features
X['age_hours'] = X['age'] * X['hours-per-week']
X['capital_ratio'] = X['capital-gain'] / (X['capital-loss'] + 1)
X['education_experience'] = X['education-num'] * X['age']

# Scale numeric features
scaler = StandardScaler()
numeric_cols = X.select_dtypes(include=[np.number]).columns
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

def train_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Model 1: XGBoost with optimal parameters for 90%+ accuracy
    model_original = XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=8,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.1,
        reg_alpha=0.05,
        reg_lambda=0.1,
        random_state=42,
        n_jobs=-1,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    model_original.fit(X_train, y_train)
    y_pred_original = model_original.predict(X_test)
    acc_original = accuracy_score(y_test, y_pred_original)
    
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model_original, X_train, y_train, cv=cv, scoring='accuracy')
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    
    # Model 2: Fair Model
    X_train_fair = X_train.drop(['race', 'sex'], axis=1)
    X_test_fair = X_test.drop(['race', 'sex'], axis=1)
    
    model_fair = XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=8,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.1,
        reg_alpha=0.05,
        reg_lambda=0.1,
        random_state=42,
        n_jobs=-1,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    model_fair.fit(X_train_fair, y_train)
    y_pred_fair = model_fair.predict(X_test_fair)
    acc_fair = accuracy_score(y_test, y_pred_fair)
    
    fairness_original = calculate_fairness(X_test, y_pred_original)
    fairness_fair = calculate_fairness(X_test_fair, y_pred_fair)
    
    return {
        'model_original': model_original,
        'model_fair': model_fair,
        'acc_original': acc_original,
        'acc_fair': acc_fair,
        'cv_mean': cv_mean,
        'cv_std': cv_std,
        'y_pred_original': y_pred_original,
        'y_pred_fair': y_pred_fair,
        'y_test': y_test,
        'X_test': X_test,
        'X_test_fair': X_test_fair,
        'fairness_original': fairness_original,
        'fairness_fair': fairness_fair
    }

def calculate_fairness(X_test, y_pred):
    fairness_metrics = {}
    sensitive_attrs = ['race', 'sex']
    
    for attr in sensitive_attrs:
        if attr in X_test.columns:
            attr_values = X_test[attr]
            grouped_data = pd.DataFrame({
                'group': attr_values,
                'prediction': y_pred
            })
            pos_rates = grouped_data.groupby('group')['prediction'].mean()
            
            if len(pos_rates) > 1:
                max_rate = pos_rates.max()
                min_rate = pos_rates.min()
                disparity = max_rate / min_rate if min_rate > 0 else float('inf')
                fairness_metrics[attr] = {
                    'disparity_ratio': disparity,
                    'positive_rates': pos_rates.to_dict()
                }
            else:
                fairness_metrics[attr] = None
        else:
            fairness_metrics[attr] = None
    
    return fairness_metrics

with st.spinner("Training XGBoost model for 90%+ accuracy..."):
    start_time = time.time()
    results = train_models(X, y)
    end_time = time.time()
    training_time = end_time - start_time

acc_original = results['acc_original'] * 100

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        "Original Model", 
        f"{acc_original:.2f}%",
        delta="✅ 90%+ Achieved!" if acc_original >= 90 else f"Need {90-acc_original:.1f}% more"
    )

with col2:
    st.metric(
        "Cross-Validation", 
        f"{results['cv_mean']*100:.2f}%", 
        f"±{results['cv_std']*100:.2f}%"
    )

with col3:
    st.metric(
        "Fair Model", 
        f"{results['acc_fair']*100:.2f}%"
    )

with col4:
    drop = (results['acc_original'] - results['acc_fair']) * 100
    st.metric(
        "Accuracy Drop", 
        f"{drop:.2f}%"
    )

with col5:
    st.metric(
        "Training Time", 
        f"{training_time:.1f}s"
    )

if acc_original >= 90:
    st.success(f"🎉 **SUCCESS!** Model achieved {acc_original:.2f}% accuracy - Target reached!")

cm = confusion_matrix(results['y_test'], results['y_pred_original'])
col1, col2, col3 = st.columns(3)
col1.metric("True Positives", cm[1][1])
col2.metric("True Negatives", cm[0][0])
col3.metric("Total Samples", len(results['y_test']))

with st.expander("📊 Original Model Classification Report"):
    report = classification_report(results['y_test'], results['y_pred_original'], output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.style.background_gradient(cmap='Blues'))

with st.expander("📊 Fair Model Classification Report"):
    report = classification_report(results['y_test'], results['y_pred_fair'], output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.style.background_gradient(cmap='Blues'))

st.subheader("⚖️ Fairness Comparison: Original vs Fair Model")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

if results['fairness_original']['race'] is not None:
    race_rates = results['fairness_original']['race']['positive_rates']
    axes[0].bar(race_rates.keys(), race_rates.values())
    axes[0].set_title('Original Model: Positive Rates by Race')
    axes[0].set_ylabel('Positive Prediction Rate')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].axhline(y=np.mean(list(race_rates.values())), color='r', linestyle='--', label='Average')
    axes[0].legend()

if results['fairness_fair']['race'] is not None:
    race_rates_fair = results['fairness_fair']['race']['positive_rates']
    axes[1].bar(race_rates_fair.keys(), race_rates_fair.values())
    axes[1].set_title('Fair Model: Positive Rates by Race')
    axes[1].set_ylabel('Positive Prediction Rate')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].axhline(y=np.mean(list(race_rates_fair.values())), color='r', linestyle='--', label='Average')
    axes[1].legend()

plt.tight_layout()
st.pyplot(fig)

st.subheader("📊 Disparity Ratio Comparison")
disparity_data = []
for model_type in ['original', 'fair']:
    fairness_data = results[f'fairness_{model_type}']
    for attr in ['race', 'sex']:
        if fairness_data[attr] is not None:
            disparity_data.append({
                'Model': 'Original' if model_type == 'original' else 'Fair',
                'Attribute': attr.capitalize(),
                'Disparity Ratio': fairness_data[attr]['disparity_ratio']
            })

if disparity_data:
    disparity_df = pd.DataFrame(disparity_data)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=disparity_df, x='Attribute', y='Disparity Ratio', hue='Model', ax=ax)
    ax.axhline(y=1.2, color='r', linestyle='--', label='Acceptable Threshold (1.2)')
    ax.set_title('Disparity Ratio Comparison')
    ax.legend()
    st.pyplot(fig)

st.subheader("📊 Feature Importance (Original Model)")

importance_df = pd.DataFrame({
    'Feature': results['X_test'].columns,
    'Importance': results['model_original'].feature_importances_
}).sort_values('Importance', ascending=False)

col1, col2 = st.columns([2, 1])

with col1:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(importance_df['Feature'][:10], importance_df['Importance'][:10])
    ax.set_xlabel('Feature Importance')
    ax.set_title('Top 10 Feature Importance (XGBoost)')
    plt.tight_layout()
    st.pyplot(fig)

with col2:
    st.dataframe(importance_df.head(10))

st.subheader("⚖️ Bias Detection Analysis")

sensitive_attrs = ['race', 'sex']

for attr in sensitive_attrs:
    if attr in data.columns and attr in X.columns:
        original_labels = data[attr]
        
        grouped_data = pd.DataFrame({
            'group': original_labels,
            'income': y
        })
        pos_rates = grouped_data.groupby('group')['income'].mean()
        
        st.write(f"### Bias Analysis by '{attr}'")
        
        display_df = pd.DataFrame({
            'Group': pos_rates.index,
            'Positive Rate (%)': (pos_rates.values * 100).round(1),
            'Count': grouped_data.groupby('group').size().values
        })
        st.dataframe(display_df)
        
        if len(pos_rates) > 1:
            max_rate = pos_rates.max()
            min_rate = pos_rates.min()
            disparity = max_rate / min_rate if min_rate > 0 else float('inf')
            st.metric(f"Disparity Ratio ({attr})", f"{disparity:.2f}x")
            
            if disparity > 1.5:
                st.warning(f"⚠️ High bias detected in '{attr}'! Disparity ratio is {disparity:.2f}x")
            elif disparity > 1.2:
                st.info(f"⚡ Moderate bias detected in '{attr}'")
            else:
                st.success(f"✅ Low bias detected in '{attr}'")

st.subheader("🔄 Counterfactual Analysis")

if st.button("🎯 Find Counterfactual Example"):
    with st.spinner("Searching for counterfactual..."):
        y_pred = results['model_original'].predict(results['X_test'])
        incorrect_indices = np.where(y_pred != results['y_test'])[0]
        
        if len(incorrect_indices) > 0:
            sample_idx = incorrect_indices[0]
            sample = results['X_test'].iloc[[sample_idx]]
            original_pred = results['model_original'].predict(sample)[0]
            actual = results['y_test'].iloc[sample_idx]
            
            st.write(f"**Sample Analysis:**")
            st.write(f"- Actual: {'>50K' if actual == 1 else '<=50K'}")
            st.write(f"- Predicted: {'>50K' if original_pred == 1 else '<=50K'}")
            st.write(f"- Status: {'❌ Incorrect' if original_pred != actual else '✅ Correct'}")
            
            counterfactual_found = False
            X_train, _, _, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            
            for feature_idx, feature_name in enumerate(results['X_test'].columns):
                if feature_name in ['race', 'sex']:
                    continue
                    
                feature_values = X_train.iloc[:, feature_idx]
                test_values = [
                    feature_values.quantile(0.25),
                    feature_values.quantile(0.50),
                    feature_values.quantile(0.75),
                    feature_values.mean(),
                    feature_values.max()
                ]
                
                for value in test_values:
                    if value == sample.iloc[0, feature_idx]:
                        continue
                        
                    modified_sample = sample.copy()
                    modified_sample.iloc[0, feature_idx] = value
                    new_pred = results['model_original'].predict(modified_sample)[0]
                    
                    if new_pred != original_pred:
                        st.success(f"✅ Counterfactual found!")
                        st.write(f"**Change '{feature_name}'** from {sample.iloc[0, feature_idx]:.2f} to {value:.2f}")
                        st.write(f"New prediction: {'>50K' if new_pred == 1 else '<=50K'}")
                        counterfactual_found = True
                        break
                
                if counterfactual_found:
                    break
            
            if not counterfactual_found:
                st.warning("No counterfactual found for this sample.")
        else:
            st.info("All samples are correctly predicted! 🎉")

st.sidebar.markdown("---")