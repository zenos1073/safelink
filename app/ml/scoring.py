import json
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse

import joblib
import pandas as pd
from flask import current_app

from app.ml.features import (
    FEATURE_COLUMNS,
    extract_url_features,
)


DISPLAY_NAMES = {
    "URLLength": "Unusually long URL",
    "DomainLength": "Long domain name",
    "IsDomainIP": "IP address used instead of a domain",
    "NoOfSubDomain": "Multiple subdomains",
    "HasObfuscation": "Encoded or obfuscated characters",
    "NoOfDegitsInURL": "Many digits in the URL",
    "NoOfEqualsInURL": "Multiple URL parameters",
    "NoOfQMarkInURL": "Query string present",
    "NoOfAmpersandInURL": "Several query parameters",
    "IsHTTPS": "No HTTPS encryption",
}


# Known brands and their official domains.
# These are used only as an additional security rule.

KNOWN_BRANDS = {
    "amazon": [
        "amazon.com",
        "amazon.in",
    ],
    "google": [
        "google.com",
        "google.in",
        "google.co.in",
    ],
    "microsoft": [
        "microsoft.com",
        "microsoft.in",
        "microsoft.co.in",
    ],
    "netflix": [
        "netflix.com",
        "netflix.in",
    ],
    "paypal": [
        "paypal.com",
    ],
    "apple": [
        "apple.com",
    ],
    "facebook": [
        "facebook.com",
    ],
    "instagram": [
        "instagram.com",
    ],
    "linkedin": [
        "linkedin.com",
    ],
    "whatsapp": [
        "whatsapp.com",
    ],
}


@lru_cache(maxsize=2)
def _load_model(model_path):
    """
    Load the trained phishing-detection model.
    """

    path = Path(model_path)

    if not path.exists():
        return None

    return joblib.load(path)


def _heuristic_probability(features):
    """
    Safe temporary scoring used when a trained model is unavailable.

    This returns a PHISHING probability.
    """

    risk = 0.08

    risk += min(
        features["URLLength"] / 400,
        0.18,
    )

    risk += (
        0.30
        if features["IsDomainIP"]
        else 0
    )

    risk += min(
        features["NoOfSubDomain"] * 0.07,
        0.21,
    )

    risk += (
        0.20
        if features["HasObfuscation"]
        else 0
    )

    risk += min(
        features["NoOfDegitsInURL"] * 0.012,
        0.12,
    )

    risk += min(
        features["NoOfEqualsInURL"] * 0.04,
        0.12,
    )

    risk += (
        0.10
        if not features["IsHTTPS"]
        else 0
    )

    return min(
        risk,
        0.95,
    )


def _get_domain(url):
    """
    Return the hostname from a URL.
    """

    parsed = urlparse(url)

    domain = parsed.netloc.lower()

    if "@" in domain:
        domain = domain.split("@")[-1]

    if ":" in domain:
        domain = domain.split(":")[0]

    return domain


def _brand_impersonation(url):
    """
    Check whether a domain appears to imitate a known brand
    without using the brand's official domain.
    """

    domain = _get_domain(url)

    if not domain:
        return None

    for brand, official_domains in KNOWN_BRANDS.items():

        # Allow the official domain and its subdomains.
        for official_domain in official_domains:

            if (
                domain == official_domain
                or domain.endswith(
                    "." + official_domain
                )
            ):
                return None

        # Remove common separators for lookalike detection.
        cleaned_domain = (
            domain
            .replace("-", "")
            .replace("_", "")
        )

        # Flag domains where the brand appears in the
        # domain name but the domain is not official.
        if brand in cleaned_domain:

            return (
                f"Possible {brand.title()} brand impersonation: "
                f"the domain resembles {brand.title()} but is not "
                f"the official domain."
            )

    return None


def _reasons(features, url):
    """
    Generate human-readable reasons for the prediction.
    """

    rules = [
        (
            features["IsDomainIP"],
            "The address uses an IP number instead of a named domain.",
        ),

        (
            features["HasObfuscation"],
            "The URL contains encoded characters often used to hide its destination.",
        ),

        (
            features["URLLength"] > 100,
            "The URL is unusually long.",
        ),

        (
            features["NoOfSubDomain"] >= 3,
            "The domain contains several nested subdomains.",
        ),

        (
            features["NoOfDegitsInURL"] >= 7,
            "The URL contains an unusually high number of digits.",
        ),

        (
            features["NoOfEqualsInURL"] >= 3,
            "The URL has multiple parameter assignments.",
        ),

        (
            not features["IsHTTPS"],
            "The URL does not use HTTPS encryption.",
        ),
    ]

    reasons = [
        message
        for condition, message in rules
        if condition
    ]

    brand_warning = _brand_impersonation(url)

    if brand_warning:
        reasons.append(brand_warning)

    return reasons


def _has_strong_url_warning(features):
    """
    Return whether the URL contains an independent structural warning.

    The training data is heavily biased toward particular TLDs and URL
    schemes.  Those values must not be used as phishing evidence on their
    own: ``.in`` is a valid public suffix and HTTPS is a positive security
    signal.  This helper lists only signals that are meaningful without
    relying on that dataset bias.
    """

    return any(
        [
            features["IsDomainIP"],
            features["HasObfuscation"],
            features["URLLength"] > 100,
            features["NoOfSubDomain"] >= 3,
            features["NoOfDegitsInURL"] >= 7,
            features["NoOfEqualsInURL"] >= 3,
        ]
    )


def score_url(value):
    """
    Analyse a URL and return its phishing-risk result.

    IMPORTANT:
    The training dataset uses:

        0 = phishing
        1 = legitimate

    Therefore predict_proba()[0][0] is the phishing
    probability.

    The displayed URL remains exactly as returned by
    extract_url_features(), while the ML features treat:

        example.com
        example.com/

    as the same URL structure.
    """

    # Extract the URL and ML features.
    url, features = extract_url_features(value)

    # Create the DataFrame using the same feature order
    # used during model training.
    frame = pd.DataFrame(
        [features],
        columns=FEATURE_COLUMNS,
    )

    # Load the trained model.
    model = _load_model(
        current_app.config["MODEL_PATH"]
    )

    if model is None:

        # No trained model available.
        probability = _heuristic_probability(
            features
        )

        source = (
            "Safety heuristic "
            "(train the model to enable ML predictions)"
        )

    else:

        # -------------------------------------------------
        # IMPORTANT CLASS MAPPING
        # -------------------------------------------------
        #
        # Dataset:
        #
        # 0 = phishing
        # 1 = legitimate
        #
        # Therefore:
        #
        # predict_proba()[0][0]
        #       = phishing probability
        #
        # predict_proba()[0][1]
        #       = legitimate probability
        #
        # -------------------------------------------------

        probabilities = model.predict_proba(frame)[0]

        probability = float(
            probabilities[0]
        )

        source = "Trained machine-learning model"

    # Generate explanation reasons.
    reasons = _reasons(
        features,
        url,
    )

    # Brand impersonation is an additional security signal.
    brand_warning = _brand_impersonation(url)

    if brand_warning:

        # A suspected brand impersonation should
        # significantly increase the phishing risk.
        probability = max(
            probability,
            0.80,
        )

    # Do not let the model's learned TLD/scheme bias classify a clean URL.
    # In particular, ``.in`` is not suspicious and ``https`` is not a
    # phishing indicator.  Keep a genuinely suspicious URL eligible for a
    # phishing result when it has independent structural evidence.
    elif (
        features["IsHTTPS"]
        and not _has_strong_url_warning(features)
    ):
        probability = min(probability, 0.20)

    # Keep the displayed probability within a
    # sensible range.
    probability = min(
        max(probability, 0.0),
        0.95,
    )

    # Determine the final verdict.
    verdict = (
        "phishing"
        if probability >= 0.50
        else "legitimate"
    )

    # Determine risk level.
    risk_level = (
        "high"
        if probability >= 0.75
        else "medium"
        if probability >= 0.50
        else "low"
    )

    # If no specific warning was found,
    # provide a neutral explanation.
    if not reasons:

        reasons = [
            "No prominent URL-structure warning signs were found."
        ]

    return {
        # The URL shown to the user.
        #
        # The original trailing slash is preserved.
        "url": url,

        # Features supplied to the ML model.
        "features": features,

        # Phishing confidence percentage.
        "confidence": round(
            probability * 100,
            1,
        ),

        # phishing / legitimate
        "verdict": verdict,

        # low / medium / high
        "risk_level": risk_level,

        # Human-readable explanations.
        "reasons": reasons,

        # Model used for the prediction.
        "source": source,
    }


def serialise_reasons(reasons):
    """
    Convert reasons into JSON for storage.
    """

    return json.dumps(reasons)


def deserialise_reasons(value):
    """
    Convert stored JSON reasons back into a Python list.
    """

    return json.loads(value)
