import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score

loan_data = pd.read_csv("/Users/ramko/Downloads/Task 3 and 4_Loan_Data.csv")

feature_columns = [
    "credit_lines_outstanding",
    "loan_amt_outstanding",
    "total_debt_outstanding",
    "income",
    "years_employed",
    "fico_score"
]
X_data = loan_data[feature_columns]
y_labels = loan_data["default"]

X_train_set, X_test_set, y_train_labels, y_test_labels = train_test_split(
    X_data, y_labels, test_size=0.2, random_state=42, stratify=y_labels
)

models_dict = {
    "logistic": Pipeline([
        ("scaler", StandardScaler()),
        ("logistic_model", LogisticRegression(max_iter=1000))
    ]),
    "decision_tree": Pipeline([
        ("scaler", StandardScaler()),
        ("tree_model", DecisionTreeClassifier(max_depth=4, random_state=42))
    ])
}

highest_auc = 0
best_model_pipeline = None
best_model_name = ""

for model_name, pipeline in models_dict.items():
    pipeline.fit(X_train_set, y_train_labels)
    y_pred_probs = pipeline.predict_proba(X_test_set)[:, 1]
    auc = roc_auc_score(y_test_labels, y_pred_probs)
    if auc > highest_auc:
        highest_auc = auc
        best_model_pipeline = pipeline
        best_model_name = model_name

joblib.dump(best_model_pipeline, "pd_pipeline.pkl")


def predict_default_probability(borrower_info: dict) -> float:
    borrower_df = pd.DataFrame([borrower_info]).reindex(columns=feature_columns, fill_value=0)
    return best_model_pipeline.predict_proba(borrower_df)[0, 1]


def calculate_expected_loss(borrower_info: dict, recovery_rate: float = 0.10) -> float:
    prob_default = predict_default_probability(borrower_info)
    exposure_at_default = borrower_info.get("loan_amt_outstanding", 0)
    return prob_default * (1 - recovery_rate) * exposure_at_default


# TEST CASE
if __name__ == "__main__":
    sample_borrower = {
        "credit_lines_outstanding": 4,
        "loan_amt_outstanding": 25000,
        "total_debt_outstanding": 40000,
        "income": 85000,
        "years_employed": 5,
        "fico_score": 690
    }
    predicted_pd = predict_default_probability(sample_borrower)
    predicted_el = calculate_expected_loss(sample_borrower)
    print("PD:", predicted_pd)
    print("EL: $", predicted_el)
