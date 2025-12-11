import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

import dagshub 

dagshub_owner = "zeinabkoubaissi33"
dagshub_repo = "accelerating-tracking-demo"
dagshub.init(repo_owner=dagshub_owner, repo_name=dagshub_repo,mlflow=True)
# Enable MLflow autologging (optional)
#mlflow.set_tracking_uri("http://127.0.0.1:5000")
#mlflow.set_experiment("demo_experiment")
  # Adjust if needed

mlflow.sklearn.autolog()
MODEL_NAME = "IrisRandomForestModel"

def train_model():
    # Load sample dataset
    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )

    # Define model parameters
    n_estimators = 100
    max_depth = 3

    # Begin MLflow run
    with mlflow.start_run():
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )
        clf.fit(X_train, y_train)

        # Predictions + metrics
        preds = clf.predict(X_test)
        acc = accuracy_score(y_test, preds)

        # Log params and metrics (autolog also does it)
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_metric("accuracy", acc)

        # Log the model manually (optional)
        mlflow.sklearn.log_model(clf, "model")

        print(f"Logged model with accuracy: {acc:.4f}")

if __name__ == "__main__":
    train_model()

