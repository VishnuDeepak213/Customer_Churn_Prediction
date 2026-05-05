import argparse
from pathlib import Path


def register_model(
    model_path: Path,
    tracking_uri: str,
    experiment_name: str,
    registered_model_name: str,
    run_name: str = 'model-registry-run',
) -> str:
    if not model_path.exists():
        raise FileNotFoundError(f'Model file not found: {model_path}')

    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_artifact(str(model_path), artifact_path='model_artifact')
        model_uri = f'runs:/{run.info.run_id}/model_artifact/{model_path.name}'

    client = MlflowClient(tracking_uri=tracking_uri)
    try:
        client.get_registered_model(registered_model_name)
    except Exception:
        client.create_registered_model(registered_model_name)

    result = client.create_model_version(
        name=registered_model_name,
        source=model_uri,
        run_id=run.info.run_id,
    )

    return f'{registered_model_name}:{result.version}'


def main() -> None:
    parser = argparse.ArgumentParser(description='Register trained model in MLflow Model Registry.')
    parser.add_argument('--model-path', default='models/best_model.pkl')
    parser.add_argument('--tracking-uri', default='http://localhost:5000')
    parser.add_argument('--experiment-name', default='churn-prediction')
    parser.add_argument('--registered-model-name', default='churn-predictor')
    parser.add_argument('--run-name', default='register-churn-model')
    args = parser.parse_args()

    version_tag = register_model(
        model_path=Path(args.model_path),
        tracking_uri=args.tracking_uri,
        experiment_name=args.experiment_name,
        registered_model_name=args.registered_model_name,
        run_name=args.run_name,
    )

    print(f'REGISTERED_MODEL={version_tag}')


if __name__ == '__main__':
    main()
