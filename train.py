import numpy as np
import pandas as pd
from src.preprocess import preprocess_pipeline
import os
import mlflow
import bentoml
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from src.common.utils import get_param_set
from src.common.metrics import rmse_cv_score
from src.common.constants import ARTIFACT_PATH, DATA_PATH, LOG_FILEPATH

if __name__ == "__main__":
    train_df = pd.read_csv("/workspaces/lgcns-bike-sharing/data/bike_sharing_train.csv")
    train_df[['season','weather' ]] = train_df[['season','weather' ]].astype(str)
    _X = train_df.drop(["datetime", 'count'], axis=1)
    y = train_df["count"]
    X = preprocess_pipeline.fit_transform(X=_X, y=y)
    print(X)
    print(y)
    # Data storage - 피처 데이터 저장
    if not os.path.exists(os.path.join(DATA_PATH, "storage")):
        os.makedirs(os.path.join(DATA_PATH, "storage"))
    X.assign(rent=y).to_csv(
        os.path.join(DATA_PATH, "storage", "bike_sharing_train_features.csv"),
        index=False,
    )



    # params_candidates = {
    #     "learning_rate": [0.01, 0.05, 0.1],
    #     "max_depth": [3, 4, 5, 6],
    #     "max_features": [1.0, 0.9, 0.8, 0.7],
    # }
    params_candidates = {
        "learning_rate": [0.01, 0.1],
        "max_depth": [3, 6],
        "max_features": [1.0, 0.7],
    }
    param_set = get_param_set(params=params_candidates)

    # Set experiment name for mlflow
    experiment_name = "new_experiment"
    mlflow.set_experiment(experiment_name=experiment_name)
    mlflow.set_tracking_uri("./mlruns")
    for i, params in enumerate(param_set):


        run_name = f"Run {i}"
        with mlflow.start_run(run_name=f"Run {i}"):
            regr = GradientBoostingRegressor(**params)
            pipeline = Pipeline(
                [("preprocessor", preprocess_pipeline), ("regr", regr)]
            )
            pipeline.fit(_X, y)

            # get evaluations scores
            score_cv = rmse_cv_score(regr, X, y)


            name = regr.__class__.__name__
            mlflow.set_tag("estimator_name", name)

            # 로깅 정보 : 파라미터 정보
            mlflow.log_params({key: regr.get_params()[key] for key in params})

            # 로깅 정보: 평가 메트릭
            mlflow.log_metrics(
                {
                    "RMSE_CV": score_cv.mean(),
                }
            )

            # 로깅 정보 : 학습 loss
            for s in regr.train_score_:
                mlflow.log_metric("Train Loss", s)

            # 모델 아티팩트 저장
            mlflow.sklearn.log_model(pipeline, "model")

            # log charts
            mlflow.log_artifact(ARTIFACT_PATH)



    # Find the best regr
    best_run_df = mlflow.search_runs(
        order_by=["metrics.RMSE_CV ASC"], max_results=1
    )

    if len(best_run_df.index) == 0:
        raise Exception(f"Found no runs for experiment '{experiment_name}'")

    best_run = mlflow.get_run(best_run_df.at[0, "run_id"])
    best_params = best_run.data.params


    best_model_uri = f"{best_run.info.artifact_uri}/model"

    # 베스트 모델을 아티팩트 폴더에 복사
    # copy_tree(best_model_uri.replace("file://", ""), ARTIFACT_PATH)

    bentoml.sklearn.save_model(
        name="house_rent",
        model=mlflow.sklearn.load_model(best_model_uri),
        signatures={"predict": {"batchable": True, "batch_dim": 0}},
        metadata=best_params,
    )






