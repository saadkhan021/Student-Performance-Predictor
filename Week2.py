import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay

# Step 1: Load the dataset
df = pd.read_csv(r'C:\Users\saadk\Desktop\Week2 intern\StudentsPerformance.csv')  # Update path as needed

# Step 2: Create binary Pass/Fail column from average score
df['average_score'] = df[['math score', 'reading score', 'writing score']].mean(axis=1)
df['Pass'] = df['average_score'].apply(lambda x: 1 if x >= 50 else 0)

# Drop original scores to focus on behavioral data
df = df.drop(['math score', 'reading score', 'writing score', 'average_score'], axis=1)

# Step 3: Encode categorical variables
df_encoded = pd.get_dummies(df, drop_first=True)

# Step 4: Train/test split
X = df_encoded.drop('Pass', axis=1)
y = df_encoded['Pass']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 6: Train models
log_model = LogisticRegression()
tree_model = DecisionTreeClassifier(random_state=42)

log_model.fit(X_train, y_train)
tree_model.fit(X_train, y_train)

# Step 7: Evaluate models
def evaluate_model(model, model_name):
    y_pred = model.predict(X_test)
    print(f"\n📌 {model_name} Results:")
    print(f"✅ Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print(f"✅ Precision: {precision_score(y_test, y_pred) * 100:.2f}%")
    print(f"✅ Recall: {recall_score(y_test, y_pred) * 100:.2f}%")

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Fail", "Pass"])
    disp.plot(cmap="Blues")
    plt.title(f"{model_name} - Confusion Matrix")
    plt.show()

    return y_pred

log_preds = evaluate_model(log_model, "Logistic Regression")
tree_preds = evaluate_model(tree_model, "Decision Tree")

# Step 8: Plot actual Pass vs Fail distribution
sns.countplot(x=y, palette="Set2")
plt.xticks([0, 1], ['Fail', 'Pass'])
plt.title('Actual Student Outcomes')
plt.ylabel('Number of Students')
plt.xlabel('Outcome')
plt.show()

# Step 9: Plot predicted outcomes
def plot_predictions(preds, model_name):
    sns.countplot(x=preds, palette="Set1")
    plt.xticks([0, 1], ['Fail', 'Pass'])
    plt.title(f'Predicted Outcomes by {model_name}')
    plt.ylabel('Number of Students')
    plt.xlabel('Prediction')
    plt.show()

plot_predictions(log_preds, "Logistic Regression")
plot_predictions(tree_preds, "Decision Tree")

# Step 10: Feature importance (Decision Tree)
importances = pd.Series(tree_model.feature_importances_, index=X.columns).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=importances.values[:10], y=importances.index[:10])
plt.title('Top 10 Important Features (Decision Tree)')
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()

# Step 11: Natural language summary of key features
print("\n  Simple Explanation ")
for feature in importances.head(5).index:
    if 'lunch_standard' in feature:
        print("- Students who eat **standard lunch** are more likely to pass.")
    elif 'test preparation course_none' in feature:
        print("- Students who did **not** complete the test preparation course tend to fail more.")
    elif 'gender_male' in feature:
        print("- **Male students** may have slightly different performance than females.")
    elif 'parental level of education_' in feature:
        level = feature.split('_')[-1]
        print(f"- Students whose parents studied up to **{level}** show performance impact.")
    elif 'race/ethnicity_' in feature:
        group = feature.split('_')[-1]
        print(f"- **Race/Ethnicity group {group}** shows varying results.")
