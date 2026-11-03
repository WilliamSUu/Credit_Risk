"""五折比较XGBoost主表baseline、当前冠军和加入内部还款行为的候选模型。"""

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
from src.installment_features import build_installment_features
from src.previous_application_features import build_previous_application_features


EXPERIMENT_NAMES = {
    "application_only": "XGBoost: Application baseline",
    "application_bureau_previous": "XGBoost + Bureau + Previous (Current champion)",
    "application_bureau_previous_installments": "XGBoost + Bureau + Previous + Installments",
}
METRICS = ["roc_auc", "pr_auc", "ks", "brier_score"]


def attach_customer_features(application: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    """安全连接客户级聚合特征，保证连接前后主表客户数量完全不变。"""
    original_rows = len(application)
    merged = application.merge(history, on=CUSTOMER_ID, how="left", validate="one_to_one")
    if len(merged) != original_rows:
        raise RuntimeError("连接还款特征后主表行数变化，请检查客户键是否唯一。")

    # 无历史记录本身是一种客户状态；has标记设为0，其余特征交给缺失值流程处理。
    for column in merged.columns:
        if column.startswith("HAS_"):
            merged[column] = merged[column].fillna(0).astype("int8")
    return merged


def summarize_results(fold_results: pd.DataFrame) -> pd.DataFrame:
    """生成均值、标准差及相对主表baseline改善幅度的直观大表。"""
    baseline_means = fold_results.loc[
        fold_results["experiment"] == "application_only", METRICS
    ].mean()
    rows = []
    for experiment in EXPERIMENT_NAMES:
        group = fold_results.loc[fold_results["experiment"] == experiment]
        row = {"experiment": experiment, "display_name": EXPERIMENT_NAMES[experiment]}
        for metric in METRICS:
            mean_value = float(group[metric].mean())
            difference = mean_value - float(baseline_means[metric])
            improvement = -difference if metric == "brier_score" else difference
            row[f"{metric}_mean"] = mean_value
            row[f"{metric}_std"] = float(group[metric].std(ddof=1))
            row[f"{metric}_improvement_vs_baseline"] = improvement
            row[f"{metric}_relative_improvement_vs_baseline_pct"] = (
                improvement / float(baseline_means[metric]) * 100
            )
        rows.append(row)
    return pd.DataFrame(rows)


def make_incremental_comparison(fold_results: pd.DataFrame) -> pd.DataFrame:
    """配对比较当前冠军相对baseline，以及还款行为相对当前冠军的边际价值。"""
    steps = [
        (
            "current_champion_vs_application",
            "application_only",
            "application_bureau_previous",
        ),
        (
            "installments_vs_current_champion",
            "application_bureau_previous",
            "application_bureau_previous_installments",
        ),
    ]
    pivot = fold_results.pivot(index="fold", columns="experiment")
    rows = []
    for step_name, reference, candidate in steps:
        for metric in METRICS:
            difference = pivot[metric][candidate] - pivot[metric][reference]
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


def save_plots(fold_results: pd.DataFrame, output_dir: Path) -> None:
    """保存四指标均值误差棒总览、逐折曲线及候选模型边际改善图。"""
    order = list(EXPERIMENT_NAMES)
    colors = ["#2474B5", "#36A269", "#E87524"]
    labels = ["Application\nbaseline", "Bureau + Previous\nchampion", "+ Installment\nbehavior"]

    figure, axes = plt.subplots(2, 2, figsize=(14, 10))
    for axis, metric in zip(axes.flat, METRICS):
        grouped = fold_results.groupby("experiment")[metric].agg(["mean", "std"])
        axis.bar(
            range(3), grouped.loc[order, "mean"], yerr=grouped.loc[order, "std"],
            capsize=5, color=colors,
        )
        axis.set_xticks(range(3), labels)
        direction = "lower is better" if metric == "brier_score" else "higher is better"
        axis.set_title(f"{metric.upper()} ({direction})")
        axis.set_ylabel("5-fold mean ± standard deviation")
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Incremental value of internal repayment behavior", fontsize=16)
    figure.tight_layout()
    figure.savefig(output_dir / "installment_metrics_overview.png", dpi=180)
    plt.close(figure)

    for metric in METRICS:
        figure, axis = plt.subplots(figsize=(9, 5))
        for color, experiment in zip(colors, order):
            data = fold_results.loc[fold_results["experiment"] == experiment].sort_values("fold")
            axis.plot(
                data["fold"], data[metric], marker="o", linewidth=2,
                color=color, label=EXPERIMENT_NAMES[experiment],
            )
        axis.set(title=f"5-Fold comparison: {metric.upper()}", xlabel="Validation fold", ylabel=metric.upper())
        axis.set_xticks(sorted(fold_results["fold"].unique()))
        axis.grid(alpha=0.25)
        axis.legend()
        figure.tight_layout()
        figure.savefig(output_dir / f"fold_comparison_{metric}.png", dpi=180)
        plt.close(figure)

    # 将四项指标统一换算为“正数代表候选模型改善”的相对百分比，便于同图比较。
    pivot = fold_results.pivot(index="fold", columns="experiment")
    relative_rows = []
    for metric in METRICS:
        reference = pivot[metric]["application_bureau_previous"]
        difference = pivot[metric]["application_bureau_previous_installments"] - reference
        signed_difference = -difference if metric == "brier_score" else difference
        for fold, value in (signed_difference / reference * 100).items():
            relative_rows.append({"fold": fold, "metric": metric, "relative_improvement_pct": value})
    relative = pd.DataFrame(relative_rows)
    relative_summary = relative.groupby("metric")["relative_improvement_pct"].agg(["mean", "std"])
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.bar(
        METRICS, relative_summary.loc[METRICS, "mean"],
        yerr=relative_summary.loc[METRICS, "std"], capsize=5, color="#E87524",
    )
    axis.axhline(0, color="black", linewidth=1)
    axis.set(
        title="Installments relative improvement over current champion",
        xlabel="Metric", ylabel="Mean relative improvement across 5 folds (%)",
    )
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "installment_relative_improvement.png", dpi=180)
    plt.close(figure)


def save_champion_decision(incremental: pd.DataFrame, output_dir: Path) -> None:
    """依据候选模型相对当前冠军的五折均值方向，保存可追溯冠军决策。"""
    candidate = incremental.loc[
        incremental["incremental_step"] == "installments_vs_current_champion"
    ]
    all_means_improved = bool((candidate["mean_improvement"] > 0).all())
    selected = (
        "application_bureau_previous_installments"
        if all_means_improved else "application_bureau_previous"
    )
    record = {
        "baseline_model": "application_only",
        "previous_champion": "application_bureau_previous",
        "candidate_model": "application_bureau_previous_installments",
        "selected_champion": selected,
        "all_four_mean_metrics_improved": all_means_improved,
        "candidate_wins_by_metric": {
            row["metric"]: int(row["candidate_wins"])
            for _, row in candidate.iterrows()
        },
        "decision_rule": "candidate must improve the 5-fold mean of all four tracked metrics",
    }
    with (output_dir / "champion_decision.json").open("w", encoding="utf-8") as file:
        json.dump(record, file, ensure_ascii=False, indent=2)


def main() -> None:
    """构建内部还款特征并在完全相同客户折上运行三组XGBoost实验。"""
    config = load_config()
    output_dir = config["output_dir"] / "installment_feature_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    application_train, application_test = load_application_data(config["data_dir"])
    bureau = aggregate_bureau(load_bureau(config["history_data_dir"]))
    previous = build_previous_application_features(config["history_data_dir"])
    installments = build_installment_features(config["history_data_dir"])

    # 当前冠军保留Bureau与Previous；候选模型只在此基础上增加还款行为。
    champion_features = bureau.merge(previous, on=CUSTOMER_ID, how="outer", validate="one_to_one")
    candidate_features = champion_features.merge(
        installments, on=CUSTOMER_ID, how="outer", validate="one_to_one"
    )
    dataset_pairs = {
        "application_only": (application_train, application_test),
        "application_bureau_previous": (
            attach_customer_features(application_train, champion_features),
            attach_customer_features(application_test, champion_features),
        ),
        "application_bureau_previous_installments": (
            attach_customer_features(application_train, candidate_features),
            attach_customer_features(application_test, candidate_features),
        ),
    }

    # 三组共享同一个StratifiedKFold产生的客户位置，形成严格配对实验。
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

            # 每折预处理只学习该折训练客户，验证客户不参与填补、缩放和编码。
            preprocessor = build_preprocessor(numeric_columns, categorical_columns)
            transformed_train = preprocessor.fit_transform(x_train)
            transformed_valid = preprocessor.transform(x_valid)

            # 模型参数与树数量完全一致，因此变化可归因于还款行为特征。
            model = build_cv_xgboost(
                config["random_state"] + fold_number, config["xgboost_cv_n_estimators"]
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
            logging.info("%s，第%s折：%s", EXPERIMENT_NAMES[experiment], fold_number, metrics)

    fold_results = pd.DataFrame(fold_rows)
    summary = summarize_results(fold_results)
    incremental = make_incremental_comparison(fold_results)
    fold_results.to_csv(output_dir / "fold_metrics.csv", index=False)
    summary.to_csv(output_dir / "installment_metrics_summary.csv", index=False)
    incremental.to_csv(output_dir / "incremental_step_comparison.csv", index=False)
    save_plots(fold_results, output_dir)
    save_champion_decision(incremental, output_dir)
    logging.info("内部还款行为实验完成，结果位于%s", output_dir)


if __name__ == "__main__":
    main()
