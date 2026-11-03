"""使用分层五折交叉验证，检验 XGBoost 相对 Logistic baseline 的优势是否稳定。"""

import json
import logging
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from src.config import load_config
from src.data import load_application_data
from src.evaluation import calculate_metrics
from src.features import build_preprocessor, make_features


# 内部名称会写入 CSV，显示名称用于图例和日志，避免两处名称不一致。
MODEL_NAMES = {
    "logistic_baseline": "Logistic Regression (Baseline)",
    "xgboost_champion": "XGBoost (Champion)",
}

# Brier Score 越低越好；其余三个模型区分指标都是越高越好。
# 单独记录方向，后面计算“冠军模型是否胜出”时就不会误判 Brier。
METRIC_DIRECTIONS = {
    "roc_auc": "higher",
    "pr_auc": "higher",
    "ks": "higher",
    "brier_score": "lower",
}


def build_cv_xgboost(random_state: int, n_estimators: int) -> xgb.XGBClassifier:
    """建立交叉验证使用的 XGBoost，参数与上一轮冠军模型保持一致。

    这里不在外层验证折上执行 early stopping。原因是外层验证折的职责是提供
    完全独立的最终评分；如果又用它选择最佳树轮数，评分会受到验证信息影响。
    n_estimators 使用上一轮独立实验已经选出的 849 棵树。
    """
    return xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        n_estimators=n_estimators,
        # 较小学习率让每棵树只修正一小部分残差，降低过拟合风险。
        learning_rate=0.03,
        max_depth=6,
        min_child_weight=20,
        # 每棵树随机抽取部分客户和特征，使不同树之间不过度相似。
        subsample=0.8,
        colsample_bytree=0.8,
        # L1、L2 正则化共同限制叶节点产生过于极端的风险分数。
        reg_alpha=0.1,
        reg_lambda=1.0,
        tree_method="hist",
        random_state=random_state,
        n_jobs=-1,
    )


def build_logistic(random_state: int) -> LogisticRegression:
    """建立正式基准模型：不加入 class weight 的逻辑回归。"""
    return LogisticRegression(
        max_iter=500,
        class_weight=None,
        solver="liblinear",
        random_state=random_state,
    )


def make_summary(fold_results: pd.DataFrame) -> pd.DataFrame:
    """把逐折结果汇总为均值、标准差、最小值和最大值。"""
    metric_columns = list(METRIC_DIRECTIONS)
    summary = (
        fold_results.groupby("model")[metric_columns]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )

    # pandas 聚合后产生两层列名；改成 roc_auc_mean 这类普通列名，方便阅读。
    summary.columns = [
        "model" if first == "model" else f"{first}_{second}"
        for first, second in summary.columns
    ]
    return summary


def make_paired_comparison(fold_results: pd.DataFrame) -> pd.DataFrame:
    """逐指标计算同一折中 XGBoost 减去 Logistic 的差值和胜出折数。"""
    rows = []
    for metric, direction in METRIC_DIRECTIONS.items():
        pivot = fold_results.pivot(index="fold", columns="model", values=metric)
        raw_difference = pivot["xgboost_champion"] - pivot["logistic_baseline"]

        # 对 Brier 取相反数，使 improvement 始终满足“正数代表 XGBoost 更好”。
        improvement = raw_difference if direction == "higher" else -raw_difference
        rows.append(
            {
                "metric": metric,
                "better_direction": direction,
                "mean_xgboost_minus_logistic": float(raw_difference.mean()),
                "mean_improvement": float(improvement.mean()),
                "improvement_std": float(improvement.std(ddof=1)),
                "xgboost_wins": int((improvement > 0).sum()),
                "ties": int((improvement == 0).sum()),
                "total_folds": int(len(improvement)),
            }
        )
    return pd.DataFrame(rows)


def save_metric_plots(fold_results: pd.DataFrame, output_dir: Path) -> None:
    """为四个核心指标分别保存逐折折线图，并保存均值误差棒总览图。"""
    folds = sorted(fold_results["fold"].unique())
    colors = {"logistic_baseline": "#2474B5", "xgboost_champion": "#E87524"}

    for metric, direction in METRIC_DIRECTIONS.items():
        figure, axis = plt.subplots(figsize=(7.5, 5))
        for model_name, model_data in fold_results.groupby("model"):
            model_data = model_data.sort_values("fold")
            axis.plot(
                model_data["fold"], model_data[metric], marker="o",
                linewidth=2, label=MODEL_NAMES[model_name], color=colors[model_name],
            )

        # 标题明确指出方向，读图时不必再查询指标定义。
        direction_text = "Higher is better" if direction == "higher" else "Lower is better"
        axis.set(
            title=f"5-Fold Stratified CV: {metric.upper()} ({direction_text})",
            xlabel="Validation fold",
            ylabel=metric.upper(),
            xticks=folds,
        )
        axis.grid(alpha=0.25)
        axis.legend()
        figure.tight_layout()
        figure.savefig(output_dir / f"fold_comparison_{metric}.png", dpi=180)
        plt.close(figure)

    # 总览图将不同指标标准化为相对 Logistic 的改善百分比，正数统一表示更好。
    relative_rows = []
    for metric, direction in METRIC_DIRECTIONS.items():
        pivot = fold_results.pivot(index="fold", columns="model", values=metric)
        raw_difference = pivot["xgboost_champion"] - pivot["logistic_baseline"]
        signed_difference = raw_difference if direction == "higher" else -raw_difference
        relative_improvement = signed_difference / pivot["logistic_baseline"] * 100
        for value in relative_improvement:
            relative_rows.append({"metric": metric, "relative_improvement_pct": value})

    relative_table = pd.DataFrame(relative_rows)
    relative_summary = relative_table.groupby("metric")["relative_improvement_pct"].agg(["mean", "std"])
    metric_order = list(METRIC_DIRECTIONS)
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.bar(
        metric_order,
        relative_summary.loc[metric_order, "mean"],
        yerr=relative_summary.loc[metric_order, "std"],
        capsize=5,
        color="#E87524",
    )
    axis.axhline(0, color="black", linewidth=1)
    axis.set(
        title="XGBoost relative improvement over Logistic across 5 folds",
        xlabel="Metric",
        ylabel="Mean relative improvement (%)",
    )
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "relative_improvement_summary.png", dpi=180)
    plt.close(figure)


def save_champion_record(
    config: Dict, paired_comparison: pd.DataFrame, output_dir: Path
) -> None:
    """根据五折实际胜出次数登记冠军状态，避免把结论预先写死。"""
    # 只有四项指标都在全部折中胜出，才把冠军状态标为“已通过交叉验证确认”。
    all_metrics_won_every_fold = bool(
        (paired_comparison["xgboost_wins"] == paired_comparison["total_folds"]).all()
    )
    champion_status = (
        "confirmed_by_stratified_cross_validation"
        if all_metrics_won_every_fold
        else "mixed_cross_validation_results_require_review"
    )
    record = {
        "baseline_model": "logistic_baseline",
        "champion_model": config["champion_model"],
        "selection_basis": "single_holdout_comparison",
        "selection_metrics": ["roc_auc", "pr_auc", "ks", "brier_score", "approval_rate_analysis"],
        "status": champion_status,
        "cross_validation_folds": config["cross_validation_folds"],
        "xgboost_n_estimators": config["xgboost_cv_n_estimators"],
        "all_metrics_won_every_fold": all_metrics_won_every_fold,
        "wins_by_metric": {
            row["metric"]: int(row["xgboost_wins"])
            for _, row in paired_comparison.iterrows()
        },
    }
    with (output_dir / "champion_model_record.json").open("w", encoding="utf-8") as file:
        json.dump(record, file, ensure_ascii=False, indent=2)


def main() -> None:
    """运行完整的分层五折比较，并把表格和图片统一保存到 outputs。"""
    config = load_config()
    output_dir = config["output_dir"] / "stratified_cross_validation"
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # 本阶段继续只用 application 主表，保证变化只来自验证方式，而不是新特征。
    train, test = load_application_data(config["data_dir"])
    features, _, labels, numeric_columns, categorical_columns = make_features(train, test)

    # StratifiedKFold 会让每折违约率尽量接近全体约 8%，适合类别不平衡任务。
    splitter = StratifiedKFold(
        n_splits=config["cross_validation_folds"],
        shuffle=True,
        random_state=config["random_state"],
    )
    fold_rows: List[Dict] = []

    for fold_number, (train_positions, valid_positions) in enumerate(
        splitter.split(features, labels), start=1
    ):
        # iloc 按分层器返回的位置取数；每名客户恰好在一个外层验证折中出现一次。
        x_train = features.iloc[train_positions]
        x_valid = features.iloc[valid_positions]
        y_train = labels.iloc[train_positions]
        y_valid = labels.iloc[valid_positions]
        logging.info(
            "第 %s/%s 折：训练 %s 条，验证 %s 条，验证违约率 %.4f",
            fold_number, config["cross_validation_folds"], len(x_train), len(x_valid), y_valid.mean(),
        )

        # 每一折都建立全新的预处理器，而且只在该折训练数据上学习中位数、均值和类别。
        # 这样验证折不会参与缺失值填补或标准化，避免预处理阶段的数据泄漏。
        preprocessor = build_preprocessor(numeric_columns, categorical_columns)
        transformed_train = preprocessor.fit_transform(x_train)
        transformed_valid = preprocessor.transform(x_valid)

        # 两个模型读取完全相同的训练矩阵和验证矩阵，比较差异来自分类器本身。
        models = {
            "logistic_baseline": build_logistic(config["random_state"] + fold_number),
            "xgboost_champion": build_cv_xgboost(
                config["random_state"] + fold_number,
                config["xgboost_cv_n_estimators"],
            ),
        }
        for model_name, model in models.items():
            model.fit(transformed_train, y_train)
            probabilities = model.predict_proba(transformed_valid)[:, 1]
            metrics = calculate_metrics(y_valid, probabilities)
            fold_rows.append(
                {
                    "fold": fold_number,
                    "model": model_name,
                    "train_size": len(x_train),
                    "validation_size": len(x_valid),
                    "validation_default_rate": float(y_valid.mean()),
                    **metrics,
                }
            )
            logging.info("第 %s 折 %s：%s", fold_number, model_name, metrics)

    # 逐折明细是最原始的证据；汇总表用于快速阅读均值与波动范围。
    fold_results = pd.DataFrame(fold_rows)
    fold_results.to_csv(output_dir / "fold_metrics.csv", index=False)
    make_summary(fold_results).to_csv(output_dir / "metrics_summary.csv", index=False)
    paired_comparison = make_paired_comparison(fold_results)
    paired_comparison.to_csv(output_dir / "paired_model_comparison.csv", index=False)

    save_metric_plots(fold_results, output_dir)
    save_champion_record(config, paired_comparison, output_dir)
    logging.info("交叉验证全部完成，结果已保存到 %s", output_dir)


if __name__ == "__main__":
    main()
