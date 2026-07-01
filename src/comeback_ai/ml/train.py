import json
from pathlib import Path

import joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from comeback_ai.config import get_settings
from comeback_ai.ml.data import FEATURES, generate_demo_data


def train(artifact_dir: Path | None = None) -> dict:
    output = artifact_dir or get_settings().artifact_dir
    output.mkdir(parents=True, exist_ok=True)
    data = generate_demo_data()
    x_train, x_test, y_train, y_test = train_test_split(
        data[FEATURES], data["at_risk"], test_size=0.25, random_state=42, stratify=data["at_risk"]
    )
    preprocessor = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline([("impute", SimpleImputer()), ("scale", StandardScaler())]),
                FEATURES,
            )
        ]
    )
    candidates = {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=250,
            max_depth=8,
            min_samples_leaf=8,
            class_weight="balanced",
            random_state=42,
        ),
    }
    results: dict[str, dict] = {}
    fitted = {}
    for name, classifier in candidates.items():
        pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])
        pipeline.fit(x_train, y_train)
        probabilities = pipeline.predict_proba(x_test)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, predictions, average="binary", zero_division=0
        )
        results[name] = {
            "roc_auc": round(float(roc_auc_score(y_test, probabilities)), 4),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1": round(float(f1), 4),
        }
        fitted[name] = pipeline

    winner = max(results, key=lambda name: results[name]["f1"])
    joblib.dump(fitted[winner], output / "risk_model.joblib")
    metadata = {
        "selected_model": winner,
        "selection_metric": "f1",
        "dataset": "synthetic_demo_data",
        "training_rows": len(x_train),
        "test_rows": len(x_test),
        "features": FEATURES,
        "metrics": results,
        "limitations": [
            "Trained on synthetic data; do not use for real student decisions.",
            "A real deployment requires consent, fairness analysis, monitoring, and human review.",
        ],
    }
    (output / "model_card.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


if __name__ == "__main__":
    card = train()
    print(json.dumps(card, indent=2))
