from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, classification_report

# Create imbalanced dataset
X, y = make_classification(
    n_samples=2000,
    n_features=20,
    n_informative=10,
    n_redundant=5,
    weights=[0.95, 0.05],
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Improved approach: use class weighting
model = RandomForestClassifier(
    class_weight="balanced",
    random_state=42
)
model.fit(X_train, y_train)

preds = model.predict(X_test)

bal_acc = balanced_accuracy_score(y_test, preds)

print("Fixed training balanced accuracy:", round(bal_acc, 4))
print("\nClassification report:")
print(classification_report(y_test, preds))
print("Fix: class weighting and better evaluation metric for imbalanced data.")
