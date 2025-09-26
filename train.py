# train.py
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib, json, os

os.makedirs("models", exist_ok=True)

X, y = load_iris(return_X_y=True, as_frame=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
clf = LogisticRegression(max_iter=200).fit(Xtr, ytr)
acc = accuracy_score(yte, clf.predict(Xte))

joblib.dump(clf, "models/iris_clf.joblib")
with open("models/metadata.json", "w", encoding="utf-8") as f:
    json.dump(
        {"target_names": load_iris().target_names.tolist(), "accuracy": acc},
        f,
        ensure_ascii=False,
        indent=2,
    )

print(f"Model saved. Accuracy={acc:.3f}")
