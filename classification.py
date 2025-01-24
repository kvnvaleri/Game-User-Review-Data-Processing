import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (accuracy_score, classification_report, 
                            confusion_matrix, roc_auc_score, roc_curve)
from sklearn.preprocessing import LabelEncoder
import seaborn as sns

df = pd.read_csv('cleaned_data_video_game_reviews.csv')
print("Missing Values Check:\n", df.isnull().sum())

ordinal_cols = ['Soundtrack Quality', 'Story Quality', 'Graphics Quality']
label_encoders = {}
for col in ordinal_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

categorical_cols = [
    'Genre', 
    'Platform', 
    'Age Group Targeted',
    'Game Mode',
    'Multiplayer',
    'Requires Special Device',
    'Developer',
    'Publisher',
    'Sentiment'
]

numeric_cols = ['Price', 'Release Year', 'Game Length (Hours)', 'Min Number of Players']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

rating_threshold = df['User Rating'].quantile(0.75)
df_encoded['High Rating'] = df_encoded['User Rating'].apply(
    lambda x: 1 if x > rating_threshold else 0
)

X = df_encoded.drop(['High Rating', 'Game Title', 'User Review Text', 'User Rating'], axis=1)
numeric_X = X.select_dtypes(include=['int64', 'float64'])
correlation_matrix = numeric_X.corrwith(df_encoded['High Rating'])
selected_features = correlation_matrix[abs(correlation_matrix) > 0.1].index.tolist()

categorical_features = [col for col in X.columns if col.startswith(tuple(categorical_cols))]
selected_features.extend(categorical_features)

X = X[selected_features]

y = df_encoded['High Rating']

param_grid = {
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5, 10],
    'criterion': ['gini', 'entropy']
}

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

clf = GridSearchCV(
    DecisionTreeClassifier(class_weight='balanced'),
    param_grid,
    cv=5,
    scoring='f1'
)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:,1]

print("\nBest Parameters:", clf.best_params_)
print(f"Model Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_proba)*100:.2f}%")

fpr, tpr, thresholds = roc_curve(y_test, y_proba)
plt.figure()
plt.plot(fpr, tpr, label=f'AUC = {roc_auc_score(y_test, y_proba):.2f}')
plt.plot([0,1], [0,1], linestyle='--')
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.savefig('roc_curve.png')
plt.close()

plt.figure(figsize=(40,20), dpi=600)
plot_tree(clf.best_estimator_,
          feature_names=X.columns, 
          filled=True, 
          class_names=['Low Rating', 'High Rating'],
          proportion=True,
          rounded=True,
          precision=2,
          fontsize=12)
plt.savefig('optimized_decision_tree.png', 
            bbox_inches='tight',
            dpi=600,
            format='png')
plt.close()

feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': clf.best_estimator_.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x='importance', y='feature', data=feature_importance.head(10), palette='viridis')
plt.title('Top 10 Business-Critical Features')
plt.xlabel('Impact on Game Rating')
plt.ylabel('Game Attributes')
plt.tight_layout()
plt.savefig('business_feature_importance.png')
plt.close()

print("\nBusiness Insights:")
for idx, row in feature_importance.head(3).iterrows():
    print(f"- {row['feature']} contributes {row['importance']*100:.1f}% to rating prediction")

df_clustered = pd.read_csv('clustered_data.csv')

merged_data = df_encoded.merge(df_clustered[['Game Title', 'Cluster']], on='Game Title')
X_merged = merged_data[X.columns]
merged_data['Prediction'] = clf.predict(X_merged)

cluster_performance = merged_data.groupby('Cluster').agg({
    'High Rating': 'mean',
    'Prediction': 'mean'
}).rename(columns={
    'High Rating': 'Actual High Rating Rate',
    'Prediction': 'Predicted High Rating Rate'
})

print("\nModel Performance Across Clusters:")
print(cluster_performance)

feature_importance.to_csv('feature_importance.csv', index=False)
print("\nFeature importance data saved to 'feature_importance.csv'")

import joblib
joblib.dump(clf.best_estimator_, 'game_rating_predictor.pkl')