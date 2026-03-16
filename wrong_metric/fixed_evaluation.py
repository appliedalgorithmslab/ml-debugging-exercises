from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score

# Create moderately imbalanced dataset
X, y = make_classification(
    n_samples=2000,
    n_features=20,
    n_informative=10,
    n_redundant=5,
    weights=[0.9, 0.1],
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

preds = model.predict(X_test)
probs = model.predict_proba(X_test)[:, 1]

auc = roc_auc_score(y_test, probs)
ap = average_precision_score(y_test, probs)

print("Fixed evaluation metrics")
print("------------------------")
print("ROC AUC:", round(auc, 4))
print("Average Precision:", round(ap, 4))
print("\nClassification report:")
print(classification_report(y_test, preds))
print("Fix: use metrics better aligned with imbalanced classification performance.")
