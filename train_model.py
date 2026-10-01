"""Train and compare five phishing URL classifiers.

The model uses only URL-lexical features. No request is made to the
submitted URLs during training or prediction.
"""

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from urllib.parse import urlparse

from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from app.ml.features import (
    FEATURE_COLUMNS,
    extract_url_features,
)


def make_feature_frame(urls):
    """Convert URLs into the numerical features used by the models."""

    return pd.DataFrame(
        [
            extract_url_features(
                url,
                max_length=None,
            )[1]
            for url in urls
        ],
        columns=FEATURE_COLUMNS,
    )


def domain_group(value):
    """Return a stable website group for leakage-resistant splitting."""

    url = str(value).strip()
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    hostname = (urlparse(url).hostname or "").lower().strip(".")
    parts = [part for part in hostname.split(".") if part]

    if len(parts) >= 2:
        return ".".join(parts[-2:])

    return hostname or url


def evaluate(
    name,
    classifier,
    x_train,
    x_test,
    y_train,
    y_test,
):
    """Train and evaluate one classifier."""

    classifier.fit(x_train, y_train)

    predictions = classifier.predict(x_test)

    # label 0 = phishing
    # label 1 = legitimate
    phishing_f1 = f1_score(
        y_test,
        predictions,
        pos_label=0,
    )

    legitimate_f1 = f1_score(
        y_test,
        predictions,
        pos_label=1,
    )

    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
    )

    phishing_precision = precision_score(
        y_test,
        predictions,
        pos_label=0,
        zero_division=0,
    )

    phishing_recall = recall_score(
        y_test,
        predictions,
        pos_label=0,
        zero_division=0,
    )

    # ROC-AUC requires a score/probability for the phishing class.
    try:
        probabilities = classifier.predict_proba(x_test)[:, 0]
        roc_auc = roc_auc_score(
            (y_test == 0).astype(int),
            probabilities,
        )
    except AttributeError:
        probabilities = None
        roc_auc = None

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1],
    )

    return {
        "name": name,
        "model": classifier,
        "accuracy": float(
            accuracy_score(
                y_test,
                predictions,
            )
        ),
        "phishing_precision": float(
            phishing_precision
        ),
        "phishing_recall": float(
            phishing_recall
        ),
        "phishing_f1": float(
            phishing_f1
        ),
        "legitimate_f1": float(
            legitimate_f1
        ),
        "macro_f1": float(
            macro_f1
        ),
        "roc_auc": (
            float(roc_auc)
            if roc_auc is not None
            else None
        ),
        "confusion_matrix": matrix.tolist(),
        "classification_report": classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=[
                "phishing",
                "legitimate",
            ],
            output_dict=True,
            zero_division=0,
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train five phishing URL classifiers."
    )

    parser.add_argument(
        "--data",
        default="training data/version1.csv",
    )

    parser.add_argument(
        "--output",
        default="models/phishing_url_model.joblib",
    )

    parser.add_argument(
        "--sample-size",
        type=int,
        default=0,
        help="Optional balanced sample size; 0 uses every cleaned row.",
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    # ---------------------------------------------------------
    # Load dataset
    # ---------------------------------------------------------

    raw = pd.read_csv(
        args.data,
        usecols=["URL", "label"],
    )

    # Remove malformed and duplicate URLs before training.  Duplicates can
    # otherwise leak into both train and test and inflate the score.
    raw["URL"] = raw["URL"].astype(str).str.strip()
    raw["label"] = pd.to_numeric(raw["label"], errors="coerce")
    raw = raw.dropna(subset=["URL", "label"])
    raw = raw[raw["URL"] != ""].drop_duplicates(subset=["URL"])
    raw["label"] = raw["label"].astype(int)

    print(
        f"Loaded {len(raw):,} rows from {args.data}"
    )

    # ---------------------------------------------------------
    # Optional balanced sampling
    # ---------------------------------------------------------

    if (
        args.sample_size
        and len(raw) > args.sample_size
    ):
        samples_per_class = min(
            raw["label"].value_counts().min(),
            args.sample_size // 2,
        )

        raw = (
            raw.groupby(
                "label",
                group_keys=False,
            )
            .sample(
                n=samples_per_class,
                random_state=args.random_state,
            )
            .reset_index(drop=True)
        )

        print(
            f"Using balanced sample of "
            f"{len(raw):,} rows."
        )

    else:
        print(
            f"Using all {len(raw):,} rows."
        )

    # ---------------------------------------------------------
    # Show class distribution
    # ---------------------------------------------------------

    print("\nClass distribution:")

    class_counts = raw["label"].value_counts().sort_index()

    for label, count in class_counts.items():

        if int(label) == 0:
            name = "phishing"
        else:
            name = "legitimate"

        print(
            f"  {label} ({name}): {count:,}"
        )

    # ---------------------------------------------------------
    # Feature extraction
    # ---------------------------------------------------------

    print("\nExtracting URL features...")

    x = make_feature_frame(
        raw["URL"]
    )

    y = raw["label"].astype(int)

    print(
        f"Extracted {len(FEATURE_COLUMNS)} features."
    )

    # ---------------------------------------------------------
    # Train/test split
    # ---------------------------------------------------------

    # Keep a website and its subdomains in one side of the split.  A random
    # row split can put near-identical URLs in both sets.
    groups = raw["URL"].map(domain_group).to_numpy()
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=args.random_state,
    )
    train_indices, test_indices = next(
        splitter.split(x, y, groups=groups)
    )
    x_train = x.iloc[train_indices]
    x_test = x.iloc[test_indices]
    y_train = y.iloc[train_indices]
    y_test = y.iloc[test_indices]

    print(
        f"\nTraining rows: {len(x_train):,}"
    )

    print(
        f"Testing rows:  {len(x_test):,}"
    )

    # ---------------------------------------------------------
    # Five candidate models
    # ---------------------------------------------------------

    candidates = [

        # 1. Linear SVM
        (
            "Linear SVM",

            Pipeline(
                [
                    (
                        "scale",
                        StandardScaler(),
                    ),

                    (
                        "model",
                        SVC(
                            kernel="linear",
                            probability=True,
                            random_state=args.random_state,
                        ),
                    ),
                ]
            ),
        ),

        # 2. Random Forest
        (
            "Random Forest",

            RandomForestClassifier(
                n_estimators=250,
                min_samples_leaf=2,
                n_jobs=-1,
                class_weight="balanced",
                random_state=args.random_state,
            ),
        ),

        # 3. Gradient Boosting
        (
            "Gradient Boosting",

            HistGradientBoostingClassifier(
                max_iter=250,
                learning_rate=0.08,
                l2_regularization=0.2,
                random_state=args.random_state,
            ),
        ),

        # 4. Logistic Regression
        (
            "Logistic Regression",

            Pipeline(
                [
                    (
                        "scale",
                        StandardScaler(),
                    ),

                    (
                        "model",
                        LogisticRegression(
                            max_iter=2000,
                            class_weight="balanced",
                            random_state=args.random_state,
                        ),
                    ),
                ]
            ),
        ),

        # 5. K-Nearest Neighbors
        (
            "K-Nearest Neighbors",

            Pipeline(
                [
                    (
                        "scale",
                        StandardScaler(),
                    ),

                    (
                        "model",
                        KNeighborsClassifier(
                            n_neighbors=7,
                            weights="distance",
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
        ),
    ]

    # ---------------------------------------------------------
    # Train and evaluate all models
    # ---------------------------------------------------------

    results = []

    print(
        "\nTraining and comparing models...\n"
    )

    for name, model in candidates:

        print(
            f"Training {name}..."
        )

        result = evaluate(
            name,
            model,
            x_train,
            x_test,
            y_train,
            y_test,
        )

        results.append(result)

        print(
            f"  Accuracy:         "
            f"{result['accuracy']:.4f}"
        )

        print(
            f"  Phishing Precision:"
            f" {result['phishing_precision']:.4f}"
        )

        print(
            f"  Phishing Recall:   "
            f"{result['phishing_recall']:.4f}"
        )

        print(
            f"  Phishing F1:       "
            f"{result['phishing_f1']:.4f}"
        )

        print(
            f"  Macro F1:          "
            f"{result['macro_f1']:.4f}"
        )

        if result["roc_auc"] is not None:

            print(
                f"  ROC-AUC:           "
                f"{result['roc_auc']:.4f}"
            )

        print()

    # ---------------------------------------------------------
    # Select best model
    # ---------------------------------------------------------
    #
    # Phishing F1 is used because phishing detection
    # is the primary purpose of SafeLink.
    # ---------------------------------------------------------

    best = max(
        results,
        key=lambda item: (
            item["phishing_f1"],
            item["macro_f1"],
        ),
    )

    print(
        "========================================"
    )

    print(
        f"Selected model: {best['name']}"
    )

    print(
        f"Phishing F1: "
        f"{best['phishing_f1']:.4f}"
    )

    print(
        f"Macro F1: "
        f"{best['macro_f1']:.4f}"
    )

    print(
        "========================================"
    )

    # ---------------------------------------------------------
    # Save model
    # ---------------------------------------------------------

    output = Path(
        args.output
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        best["model"],
        output,
    )

    print(
        f"\nSaved {best['name']} to:"
    )

    print(
        output
    )

    # ---------------------------------------------------------
    # Save metrics
    # ---------------------------------------------------------

    report = {
        "dataset_rows": int(len(raw)),

        "training_rows": int(
            len(x_train)
        ),

        "testing_rows": int(
            len(x_test)
        ),

        "features": FEATURE_COLUMNS,

        "label_mapping": {
            "0": "phishing",
            "1": "legitimate",
        },

        "selected_model": best["name"],

        "selection_metric": (
            "phishing_f1"
        ),

        "results": [
            {
                key: value
                for key, value in result.items()
                if key not in {
                    "model",
                    "classification_report",
                }
            }
            for result in results
        ],

        "classification_report": (
            best["classification_report"]
        ),
    }

    report_path = output.with_suffix(
        ".metrics.json"
    )

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"Metrics saved to:"
    )

    print(
        report_path
    )


if __name__ == "__main__":
    main()
