"""
Internal Process Ticket Classifier
=================================

This script trains and evaluates a text classification model on a set of
synthetic internal‑process tickets relevant to large finance or insurance
organizations. Each ticket describes a software–team task (e.g., bug fix,
compliance update, performance improvement) and is assigned one of nine
categories.  The goal is to automatically identify the ticket type using
the textual description, which can help triage and prioritize work inside
regulated enterprises where software teams process large numbers of
tickets.

Although the long‑context, open‑weight model **gpt‑oss‑120b** from
OpenAI has been released to support advanced reasoning and tool use
with open weights【180865669010467†L20-L33】, running such a large model
requires significant hardware and may not be feasible in this environment.
Therefore this script demonstrates a traditional machine‑learning
approach based on TF–IDF features and a logistic regression classifier.
It provides a baseline for ticket classification that can be compared
against future experiments using gpt‑oss‑20b or gpt‑oss‑120b through
instruction‑based prompting.

Usage::

    python internal_process_classifier.py --input internal_process_tickets.csv

The script outputs overall accuracy, a classification report, and
writes predicted labels into ``predictions.csv``.
"""

import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import warnings


def load_data(csv_path: str):
    """Load the dataset of tickets.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file containing ``ticket`` and ``category`` columns.

    Returns
    -------
    tuple
        A tuple (tickets, labels) where ``tickets`` is a list of ticket
        texts and ``labels`` is a list of category labels.
    """
    df = pd.read_csv(csv_path)
    if "ticket" not in df.columns or "category" not in df.columns:
        raise ValueError("CSV must contain 'ticket' and 'category' columns")
    tickets = df["ticket"].astype(str).tolist()
    labels = df["category"].astype(str).tolist()
    return tickets, labels


def train_classifier(tickets, labels, test_size=0.25, random_state=42):
    """Train a logistic regression classifier on the ticket data.

    This function splits the data into training and test sets,
    vectorizes the text using TF–IDF, fits a logistic regression model,
    and evaluates it on the hold‑out set.

    Parameters
    ----------
    tickets : list[str]
        List of ticket descriptions.
    labels : list[str]
        List of category labels corresponding to the tickets.
    test_size : float, default=0.25
        Proportion of the data to use for the test set.
    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    tuple
        A tuple containing (vectorizer, model, X_test, y_test, y_pred).
    """
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        tickets, labels, test_size=test_size, stratify=labels, random_state=random_state
    )

    # Vectorize text using TF–IDF
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Train logistic regression classifier
    # Use 'lbfgs' solver and multi_class='multinomial' for multiclass classification
    model = LogisticRegression(max_iter=1000, solver="lbfgs", multi_class="multinomial")
    model.fit(X_train_vec, y_train)

    # Evaluate on test set
    y_pred = model.predict(X_test_vec)

    return vectorizer, model, X_test, y_test, y_pred


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate ticket classifier")
    parser.add_argument(
        "--input",
        type=str,
        default="internal_process_tickets.csv",
        help="CSV file containing tickets and labels",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="predictions.csv",
        help="File to write predictions (ticket, actual, predicted)",
    )
    args = parser.parse_args()

    # Load data
    tickets, labels = load_data(args.input)

    # Train and evaluate classifier
    vectorizer, model, X_test, y_test, y_pred = train_classifier(tickets, labels)

    # Print metrics
    print("Accuracy: {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))
    print("\nClassification Report:\n")
    # Suppress undefined metric warnings (may occur if a label has no samples in test)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        print(classification_report(y_test, y_pred))

    # Save predictions to CSV
    df_pred = pd.DataFrame({
        "ticket": X_test,
        "actual_label": y_test,
        "predicted_label": y_pred,
    })
    df_pred.to_csv(args.output, index=False)
    print(f"Predictions saved to {args.output}")

    # Note: For demonstration only, a few example predictions are printed.
    for idx in range(min(3, len(df_pred))):
        record = df_pred.iloc[idx]
        print(f"Example {idx+1}:\n  Ticket: {record['ticket']}\n  Actual: {record['actual_label']}\n  Predicted: {record['predicted_label']}\n")


if __name__ == "__main__":
    main()