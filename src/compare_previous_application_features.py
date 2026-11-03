"""五折比较主表 baseline、Bureau 冠军及加入 Previous Application 的候选模型。"""

import json
import logging
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.bureau_features import CUSTOMER_ID, aggregate_bureau, load_bureau
from src.config import load_config
from src.cross_validate_champion import build_cv_xgboost
from src.data import load_application_data
from src.evaluation import calculate_metrics
from src.features import build_preprocessor, make_features
from src.previous_application_features import build_previous_application_features


EXPERIMENT_NAMES = {
    "application_only": "XGBoost: Application baseline",
    "application_bureau": "XGBoost + Bureau (Current champion)",
    "application_bureau_previous": "XGBoost + Bureau + Previous Application",
}
METRICS = ["roc_auc", "pr_auc", "ks", "brier_score"]


def attach_customer_features(
    application: pd.DataFrame, customer_features: pd.DataFrame
) -> pd.DataFrame:
    """按客户安全左连接聚合特征，并确保一名申请人仍然只占一行。"""
    original_rows = len(application)
    merged = application.merge(
        customer_features, on=CUSTOMER_ID, how="left", validate="one_to_one"
    )
    if len(merged) != original_rows:
        raise RuntimeError("连接历史特征后客户行数发生变化，请检查聚合键是否唯一。")

    # 无历史记录是一个独立业务状态；显式标记为 0，其余统计值继续保留缺失。
    history_flags = [column for column in merged if column.startswith("HAS_")]
    for column in history_flags:
        merged[column] = merged[column].fillna(0).astype("int8")
    return merged


def summarize_results(fold_results: pd.DataFrame) -> pd.DataFrame:
    """生成三模型五折均值、标准差，以及相对主表 baseline 的改善大表。"""
    baseline_means = (
        fold_results.loc[fold_results["experiment"] == "application_only", METRICS]
        .mean()
    )
    rows = []
    for experiment in EXPERIMENT_NAMES:
        group = fold_results.loc[fold_results["experiment"] == experiment]
        row = {"experiment": experiment, "display_name": EXPERIMENT_NAMES[experiment]}
        for metric in METRICS:
            mean_value = float(group[metric].mean())
            raw_difference = mean_value - float(baseline_means[metric])
            improvement = -raw_difference if metric == "brier_score" else raw_difference
            row[f"{metric}_mean"] = mean_value
            row[f"{metric}_std"] = float(group[metric].std(ddof=1))
            row[f"{metric}_improvement_vs_baseline"] = improvement
            row[f"{metric}_relative_improvement_vs_baseline_pct"] = (
                improvement / float(baseline_means[metric]) * 100
            )
        rows.append(row)
    return pd.DataFrame(rows)


def make_incremental_comparison(fold_results: pd.DataFrame) -> pd.DataFrame:
    """分别检验 Bureau 的贡献及 Previous Application 对冠军模型的边际贡献。"""
    steps = [
        ("bureau_vs_application", "application_only", "application_bureau"),
        (
            "previous_vs_current_champion",
            "application_bureau",
            "application_bureau_previous",
        ),
    ]
    pivot = fold_results.pivot(index="fold", columns="experiment")
    rows = []
    for step_name, reference, candidate in steps:
        for metric in METRICS:
            raw_difference = pivot[metric][candidate] - pivot[metric][reference]
            improvement = -raw_difference if metric == "brier_score" else raw_difference
            rows.append(
                {
                    "incremental_step": step_name,
                    "reference_experiment": reference,
                    "candidate_experiment": candidate,
                    "metric": metric,
                    "mean_raw_difference": float(raw_difference.mean()),
                    "mean_improvement": float(improvement.mean()),
                    "improvement_std": float(improvement.std(ddof=1)),
                    "candidate_wins": int((improvement > 0).sum()),
                    "ties": int((improvement == 0).sum()),
                    "total_folds": int(len(improvement)),
                }
            )
    return pd.DataFrame(rows)


def save_plots(fold_results: pd.DataFrame, output_dir: Path) -> None:
    """保存四指标均值总览与逐折曲线，直观看出提升大小和稳定性。"""
    order = list(EXPERIMENT_NAMES)
    colors = ["#2474B5", "#36A269", "#E87524"]
    short_labels = ["Application\nbaseline", "+ Bureau\nchampion", "+ Previous\nApplication"]

    figure, axes = plt.subplots(2, 2, figsize=(14, 10))
    for axis, metric in zip(axes.flat, METRICS):
        grouped = fold_results.groupby("experiment")[metric].agg(["mean", "std"])
        axis.bar(
            range(3), grouped.loc[order, "mean"],
            yerr=grouped.loc[order, "std"], capsize=5, color=colors,
        )
        axis.set_xticks(range(3), short_labels)
        direction = "lower is better" if metric == "brier_score" else "higher is better"
        axis.set_title(f"{metric.upper()} ({direction})")
        axis.set_ylabel("5-fold mean ± standard deviation")
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Incremental value of Previous Application history", fontsize=16)
    figure.tight_layout()
    figure.savefig(output_dir / "previous_application_metrics_overview.png", dpi=180)
    plt.close(figure)

    for metric in METRICS:
        figure, axis = plt.subplots(figsize=(8.5, 5))
        for color, experiment in zip(colors, order):
            data = fold_results.loc[fold_results["experiment"] == experiment].sort_values("fold")
            axis.plot(
                data["fold"], data[metric], marker="o", linewidth=2,
                color=color, label=EXPERIMENT_NAMES[experiment],
            )
        axis.set(
            title=f"5-Fold comparison: {metric.upper()}",
            xlabel="Validation fold", ylabel=metric.upper(),
        )
        axis.set_xticks(sorted(fold_results["fold"].unique()))
        axis.grid(alpha=0.25)
        axis.legend()
        figure.tight_layout()
        figure.savefig(output_dir / f"fold_comparison_{metric}.png", dpi=180)
        plt.close(figure)


def save_champion_decision(
    incremental: pd.DataFrame, output_dir: Path
) -> None:
    """根据候选模型相对当前冠军的四项均值方向，记录是否更新冠军。"""
    candidate_rows = incremental.loc[
        incremental["incremental_step"] == "previous_vs_current_champion"
    ]
    all_mean_metrics_improved = bool((candidate_rows["mean_improvement"] > 0).all())
    selected_model = (
        "application_bureau_previous"
        if all_mean_metrics_improved
        else "application_bureau"
    )
    record = {
        "baseline_model": "application_only",
        "previous_champion": "application_bureau",
        "candidate_model": "application_bureau_previous",
        "selected_champion": selected_model,
        "all_four_mean_metrics_improved": all_mean_metrics_improved,
        "candidate_wins_by_metric": {
            row["metric"]: int(row["candidate_wins"])
            for _, row in candidate_rows.iterrows()
        },
        "decision_rule": "candidate must improve the 5-fold mean of all four tracked metrics",
    }
    with (output_dir / "champion_decision.json").open("w", encoding="utf-8") as file:
        json.dump(record, file, ensure_ascii=False, indent=2)


def main() -> None:
    """构建 Previous Application 特征并执行三组完全配对的分层五折实验。"""
    config = load_config()
    output_dir = config["output_dir"] / "previous_application_feature_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    application_train, application_test = load_application_data(config["data_dir"])

    # 本轮只读取 Bureau 贷款表，不读取已证明无边际提升的 Bureau Balance 月度表。
    bureau_features = aggregate_bureau(load_bureau(config["history_data_dir"]))
    previous_features = build_previous_application_features(config["history_data_dir"])
    bureau_previous_features = bureau_features.merge(
        previous_features, on=CUSTOMER_ID, how="outer", validate="one_to_one"
    )

    dataset_pairs = {
        "application_only": (application_train, application_test),
        "application_bureau": (
            attach_customer_features(application_train, bureau_features),
            attach_customer_features(application_test, bureau_features),
        ),
        "application_bureau_previous": (
            attach_customer_features(application_train, bureau_previous_features),
            attach_customer_features(application_test, bureau_previous_features),
        ),
    }

    # 折号只生成一次，三组模型在每折面对完全相同的训练客户和验证客户。
    labels = application_train["TARGET"].astype("int8")
    splitter = StratifiedKFold(
        n_splits=config["cross_validation_folds"],
        shuffle=True,
        random_state=config["random_state"],
    )
    folds = list(splitter.split(application_train, labels))
    fold_rows: List[Dict] = []

    for experiment, (train_table, test_table) in dataset_pairs.items():
        features, _, experiment_labels, numeric_columns, categorical_columns = make_features(
            train_table, test_table
        )
        logging.info("开始 %s：原始输入特征 %s 个", EXPERIMENT_NAMES[experiment], features.shape[1])

        for fold_number, (train_positions, valid_positions) in enumerate(folds, start=1):
            x_train = features.iloc[train_positions]
            x_valid = features.iloc[valid_positions]
            y_train = experiment_labels.iloc[train_positions]
            y_valid = experiment_labels.iloc[valid_positions]

            # 每折单独拟合预处理器，验证客户不参与中位数、标准差或类别编码的学习。
            preprocessor = build_preprocessor(numeric_columns, categorical_columns)
            transformed_train = preprocessor.fit_transform(x_train)
            transformed_valid = preprocessor.transform(x_valid)

            # 三组使用同一模型参数；因此指标差异可以归因于新增历史特征。
            model = build_cv_xgboost(
                config["random_state"] + fold_number,
                config["xgboost_cv_n_estimators"],
            )
            model.fit(transformed_train, y_train)
            probabilities = model.predict_proba(transformed_valid)[:, 1]
            metrics = calculate_metrics(y_valid, probabilities)
            fold_rows.append(
                {
                    "experiment": experiment,
                    "display_name": EXPERIMENT_NAMES[experiment],
                    "fold": fold_number,
                    "input_feature_count": features.shape[1],
                    "transformed_feature_count": transformed_train.shape[1],
                    "train_size": len(train_positions),
                    "validation_size": len(valid_positions),
                    **metrics,
                }
            )
            logging.info("%s，第 %s 折：%s", EXPERIMENT_NAMES[experiment], fold_number, metrics)

    fold_results = pd.DataFrame(fold_rows)
    summary = summarize_results(fold_results)
    incremental = make_incremental_comparison(fold_results)
    fold_results.to_csv(output_dir / "fold_metrics.csv", index=False)
    summary.to_csv(output_dir / "previous_application_metrics_summary.csv", index=False)
    incremental.to_csv(output_dir / "incremental_step_comparison.csv", index=False)
    save_plots(fold_results, output_dir)
    save_champion_decision(incremental, output_dir)
    logging.info("Previous Application 增量实验完成，结果位于 %s", output_dir)


if __name__ == "__main__":
    main()
