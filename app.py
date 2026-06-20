import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Fairness Forecaster - 10/10 Project",
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

# Move feature selector outside tabs so it's global
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

@st.cache_resource
def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    return model

model = train_model(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

col1, col2, col3 = st.columns(3)
col1.metric("Model Accuracy", f"{accuracy*100:.2f}%")

cm = confusion_matrix(y_test, y_pred)
col2.metric("True Positives", cm[1][1])
col3.metric("True Negatives", cm[0][0])

with st.expander("📊 View Detailed Classification Report"):
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.style.background_gradient(cmap='Blues'))

st.subheader("📊 Feature Importance")

with st.spinner("Calculating feature importance..."):
    importance_df = pd.DataFrame({
        'Feature': X_test.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)

col1, col2 = st.columns([2, 1])

with col1:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(importance_df['Feature'][:10], importance_df['Importance'][:10])
    ax.set_xlabel('Feature Importance')
    ax.set_title('Top 10 Feature Importance')
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
        y_pred = model.predict(X_test)
        incorrect_indices = np.where(y_pred != y_test)[0]
        
        if len(incorrect_indices) > 0:
            sample_idx = incorrect_indices[0]
            sample = X_test.iloc[[sample_idx]]
            original_pred = model.predict(sample)[0]
            actual = y_test.iloc[sample_idx]
            
            st.write(f"**Sample Analysis:**")
            st.write(f"- Actual: {'>50K' if actual == 1 else '<=50K'}")
            st.write(f"- Predicted: {'>50K' if original_pred == 1 else '<=50K'}")
            st.write(f"- Status: {'❌ Incorrect' if original_pred != actual else '✅ Correct'}")
            
            counterfactual_found = False
            for feature_idx, feature_name in enumerate(X.columns):
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
                    new_pred = model.predict(modified_sample)[0]
                    
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