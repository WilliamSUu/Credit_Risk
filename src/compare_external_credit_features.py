"""用相同分层五折比较主表、Bureau、Bureau+Balance 三套 XGBoost 特征。"""

import logging
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.bureau_features import CUSTOMER_ID, build_external_credit_feature_sets
from src.config import load_config
from src.cross_validate_champion import build_cv_xgboost
from src.data import load_application_data
from src.evaluation import calculate_metrics
from src.features import build_preprocessor, make_features


# 三组实验只逐步增加外部征信信息；模型算法和参数完全相同。
EXPERIMENT_NAMES = {
    "application_only": "XGBoost: Application only",
    "application_bureau": "XGBoost + Bureau",
    "application_bureau_balance": "XGBoost + Bureau + Bureau Balance",
}
METRICS = ["roc_auc", "pr_auc", "ks", "brier_score"]


def attach_history(application: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    """把一名客户一行的历史聚合特征安全连接到 application 主表。"""
    original_rows = len(application)
    merged = application.merge(history, on=CUSTOMER_ID, how="left", validate="one_to_one")
    if len(merged) != original_rows:
        raise RuntimeError("历史特征连接后主表行数发生变化，可能存在重复客户键。")

    # 没有历史记录不等于数值为 0：has 标记设为 0，其余聚合值保留缺失。
    # 后续预处理器会增加缺失指示变量，让模型区分“没有记录”和“数值较低”。
    for column in merged.columns:
        if column.startswith("HAS_BUREAU"):
            merged[column] = merged[column].fillna(0).astype("int8")
    return merged


def summarize_results(fold_results: pd.DataFrame) -> pd.DataFrame:
    """生成适合直接阅读的大表：五折均值、标准差及相对当前冠军的差值。"""
    rows = []
    baseline = fold_results.loc[fold_results["experiment"] == "application_only"]
    baseline_means = baseline[METRICS].mean()

    for experiment, group in fold_results.groupby("experiment", sort=False):
        row = {"experiment": experiment, "display_name": EXPERIMENT_NAMES[experiment]}
        for metric in METRICS:
            mean_value = float(group[metric].mean())
            std_value = float(group[metric].std(ddof=1))
            raw_delta = mean_value - float(baseline_means[metric])

            # Brier 越低越好，因此 improvement 取相反数；其余指标正差值代表提升。
            improvement = -raw_delta if metric == "brier_score" else raw_delta
            relative_improvement = improvement / float(baseline_means[metric]) * 100
            row[f"{metric}_mean"] = mean_value
            row[f"{metric}_std"] = std_value
            row[f"{metric}_improvement"] = improvement
            row[f"{metric}_relative_improvement_pct"] = relative_improvement
        rows.append(row)
    return pd.DataFrame(rows)


def make_fold_win_table(fold_results: pd.DataFrame) -> pd.DataFrame:
    """统计新增特征在相同验证折上优于 application-only 的次数。"""
    rows = []
    for experiment in ("application_bureau", "application_bureau_balance"):
        for metric in METRICS:
            pivot = fold_results.pivot(index="fold", columns="experiment", values=metric)
            difference = pivot[experiment] - pivot["application_only"]
            improvement = -difference if metric == "brier_score" else difference
            rows.append(
                {
                    "experiment": experiment,
                    "metric": metric,
                    "mean_improvement": float(improvement.mean()),
                    "improvement_std": float(improvement.std(ddof=1)),
                    "wins_vs_application_only": int((improvement > 0).sum()),
                    "ties": int((improvement == 0).sum()),
                    "total_folds": int(len(improvement)),
                }
            )
    return pd.DataFrame(rows)


def make_incremental_step_comparison(fold_results: pd.DataFrame) -> pd.DataFrame:
    """分别量化 Bureau 相对主表、Balance 相对 Bureau 的边际贡献。"""
    steps = [
        ("bureau_vs_application", "application_only", "application_bureau"),
        ("bureau_balance_vs_bureau", "application_bureau", "application_bureau_balance"),
    ]
    rows = []
    pivot_source = fold_results.pivot(index="fold", columns="experiment")
    for step_name, reference, candidate in steps:
        for metric in METRICS:
            difference = pivot_source[metric][candidate] - pivot_source[metric][reference]
            improvement = -difference if metric == "brier_score" else difference
            rows.append(
                {
                    "incremental_step": step_name,
                    "reference_experiment": reference,
                    "candidate_experiment": candidate,
                    "metric": metric,
                    "mean_raw_difference": float(difference.mean()),
                    "mean_improvement": float(improvement.mean()),
                    "improvement_std": float(improvement.std(ddof=1)),
                    "candidate_wins": int((improvement > 0).sum()),
                    "ties": int((improvement == 0).sum()),
                    "total_folds": int(len(improvement)),
                }
            )
    return pd.DataFrame(rows)


def save_comparison_plots(fold_results: pd.DataFrame, output_dir: Path) -> None:
    """保存四指标均值误差棒大图，以及逐折 ROC-AUC/KS 对比图。"""
    colors = ["#2474B5", "#36A269", "#E87524"]
    experiment_order = list(EXPERIMENT_NAMES)

    # 2×2 面板让四项指标同时可见；误差棒表示五折标准差。
    figure, axes = plt.subplots(2, 2, figsize=(14, 10))
    for axis, metric in zip(axes.flat, METRICS):
        grouped = fold_results.groupby("experiment")[metric].agg(["mean", "std"])
        means = grouped.loc[experiment_order, "mean"]
        stds = grouped.loc[experiment_order, "std"]
        axis.bar(range(3), means, yerr=stds, capsize=5, color=colors)
        axis.set_xticks(range(3), ["Application", "+ Bureau", "+ Bureau\n+ Balance"])
        direction = "lower is better" if metric == "brier_score" else "higher is better"
        axis.set_title(f"{metric.upper()} ({direction})")
        axis.set_ylabel("5-fold mean ± standard deviation")
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Incremental value of external credit history", fontsize=16)
    figure.tight_layout()
    figure.savefig(output_dir / "external_credit_metrics_overview.png", dpi=180)
    plt.close(figure)

    # 逐折曲线用于检查提升是否稳定，避免只看均值掩盖某些折性能下降。
    for metric in ("roc_auc", "pr_auc", "ks", "brier_score"):
        figure, axis = plt.subplots(figsize=(8, 5))
        for color, experiment in zip(colors, experiment_order):
            data = fold_results.loc[fold_results["experiment"] == experiment].sort_values("fold")
            axis.plot(data["fold"], data[metric], marker="o", linewidth=2, color=color,
                      label=EXPERIMENT_NAMES[experiment])
        axis.set(title=f"5-Fold comparison: {metric.upper()}", xlabel="Validation fold", ylabel=metric.upper())
        axis.set_xticks(sorted(fold_results["fold"].unique()))
        axis.grid(alpha=0.25)
        axis.legend()
        figure.tight_layout()
        figure.savefig(output_dir / f"fold_comparison_{metric}.png", dpi=180)
        plt.close(figure)


def main() -> None:
    """构建两级外部征信特征，并在完全相同的五折上执行三组增量实验。"""
    config = load_config()
    output_dir = config["output_dir"] / "external_credit_feature_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    application_train, application_test = load_application_data(config["data_dir"])
    bureau_features, bureau_balance_features = build_external_credit_feature_sets(
        config["history_data_dir"]
    )

    # 训练表和 Kaggle 测试表都连接同一套无标签历史聚合，确保字段结构保持一致。
    dataset_pairs = {
        "application_only": (application_train, application_test),
        "application_bureau": (
            attach_history(application_train, bureau_features),
            attach_history(application_test, bureau_features),
        ),
        "application_bureau_balance": (
            attach_history(application_train, bureau_balance_features),
            attach_history(application_test, bureau_balance_features),
        ),
    }

    # 只用主表标签生成一次折号，三组实验随后共享这些客户位置，实现严格配对比较。
    labels = application_train["TARGET"].astype("int8")
    splitter = StratifiedKFold(
        n_splits=config["cross_validation_folds"], shuffle=True, random_state=config["random_state"]
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

            # 每个“实验×折”都建立独立预处理器，只从该折训练客户学习填补和缩放规则。
            preprocessor = build_preprocessor(numeric_columns, categorical_columns)
            transformed_train = preprocessor.fit_transform(x_train)
            transformed_valid = preprocessor.transform(x_valid)

            # 使用已由上一轮 early stopping 确定的 849 棵树；外层验证折只负责最终评分。
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
    win_table = make_fold_win_table(fold_results)
    incremental_table = make_incremental_step_comparison(fold_results)
    fold_results.to_csv(output_dir / "fold_metrics.csv", index=False)
    summary.to_csv(output_dir / "external_credit_metrics_summary.csv", index=False)
    win_table.to_csv(output_dir / "fold_wins_vs_application_only.csv", index=False)
    incremental_table.to_csv(output_dir / "incremental_step_comparison.csv", index=False)
    save_comparison_plots(fold_results, output_dir)
    logging.info("外部征信增量实验全部完成，结果位于 %s", output_dir)


if __name__ == "__main__":
    main()
