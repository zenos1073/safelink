import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from app.ml.features import FEATURE_COLUMNS, extract_url_features


def make_feature_frame(urls):
    return pd.DataFrame(
        [extract_url_features(url, max_length=None)[1] for url in urls],
        columns=FEATURE_COLUMNS,
    )


def evaluate(name, classifier, x_train, x_test, y_train, y_test):
    classifier.fit(x_train, y_train)
    predictions = classifier.predict(x_test)
    return {
        "name": name,
        "model": classifier,
        "accuracy": float(accuracy_score(y_test, predictions)),
        "f1": float(f1_score(y_test, predictions)),
    }


def main():
    parser = argparse.ArgumentParser(description="Train a phishing URL model.")
    parser.add_argument("--data", default="training data/version1.csv")
    parser.add_argument("--output", default="models/phishing_url_model.joblib")
    parser.add_argument("--sample-size", type=int, default=60000, help="0 uses every row.")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    raw = pd.read_csv(args.data, usecols=["URL", "label"])
    if args.sample_size and len(raw) > args.sample_size:
        raw = raw.groupby("label", group_keys=False).sample(
            n=min(raw["label"].value_counts().min(), args.sample_size // 2),
            random_state=args.random_state,
        )

    x = make_feature_frame(raw["URL"])
    y = raw["label"].astype(int)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=args.random_state
    )

    candidates = [
        (
            "Linear SVM",
            Pipeline(
                [
                    ("scale", StandardScaler()),
                    (
                        "model",
                        CalibratedClassifierCV(
                            SVC(kernel="linear", random_state=args.random_state),
                            cv=3,
                            ensemble=False,
                        ),
                    ),
                ]
            ),
        ),
        ("Random Forest", RandomForestClassifier(n_estimators=250, min_samples_leaf=2, n_jobs=-1, class_weight="balanced", random_state=args.random_state)),
        ("Gradient Boosting", HistGradientBoostingClassifier(max_iter=250, learning_rate=0.08, l2_regularization=0.2, random_state=args.random_state)),
    ]
    results = [evaluate(name, model, x_train, x_test, y_train, y_test) for name, model in candidates]
    best = max(results, key=lambda item: item["f1"])

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best["model"], output)
    report = {
        "dataset_rows": int(len(raw)),
        "features": FEATURE_COLUMNS,
        "selected_model": best["name"],
        "results": [{key: value for key, value in item.items() if key != "model"} for item in results],
        "classification_report": classification_report(y_test, best["model"].predict(x_test), output_dict=True),
    }
    report_path = output.with_suffix(".metrics.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Saved {best['name']} to {output}")
    for result in results:
        print(f"{result['name']}: accuracy={result['accuracy']:.3f}, F1={result['f1']:.3f}")
    print(f"Metrics: {report_path}")


if __name__ == "__main__":
    main()
