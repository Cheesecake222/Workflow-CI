import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


mlflow.set_experiment("heart-failure-ci")

train_df = pd.read_csv("heart_preprocessing/heart_train.csv")
test_df  = pd.read_csv("heart_preprocessing/heart_test.csv")

X_train = train_df.drop("HeartDisease", axis=1)
y_train = train_df["HeartDisease"]
X_test  = test_df.drop("HeartDisease", axis=1)
y_test  = test_df["HeartDisease"]

mlflow.sklearn.autolog()

with mlflow.start_run(run_name="RandomForest_CI") as run:
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
    mlflow.log_metric("precision", precision_score(y_test, y_pred))
    mlflow.log_metric("recall", recall_score(y_test, y_pred))
    mlflow.log_metric("f1_score", f1_score(y_test, y_pred))
    mlflow.log_metric("roc_auc", roc_auc_score(y_test, y_prob))

    print(f"Run ID  : {run.info.run_id}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
    print(f"ROC AUC : {roc_auc_score(y_test, y_prob):.4f}")
    print("Training selesai!")
