"""在完全相同的数据上比较带类别权重与不带类别权重的逻辑回归。"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.config import load_config
from src.data import ID_COLUMN, load_application_data
from src.evaluation import calculate_metrics, make_threshold_table
from src.features import build_preprocessor, make_features


# 两个模型只有 class_weight 不同，其余设置必须保持相同，才能进行公平比较。
# balanced：提高少数类（违约客户）在训练损失中的权重。
# unbalanced：使用原始样本比例，每条记录对训练损失的权重相同。
MODEL_SETTINGS: Dict[str, Optional[str]] = {
    "balanced": "balanced",
    "unbalanced": None,
}


def build_logistic_model(
    numeric_columns: list,
    categorical_columns: list,
    class_weight: Optional[str],
    random_state: int,
) -> Pipeline:
    """建立一个完整流水线，类别权重是两个实验之间唯一变化的参数。"""
    # 每个模型都拥有自己的预处理器，防止某个模型复用另一个模型的拟合状态。
    preprocessor = build_preprocessor(numeric_columns, categorical_columns)

    # class_weight=None 表示每条样本权重相同；"balanced" 按类别频率反向加权。
    classifier = LogisticRegression(
        max_iter=500,
        class_weight=class_weight,
        solver="liblinear",
        random_state=random_state,
    )

    # Pipeline 保证中位数、均值、独热编码类别和回归系数只从训练集学习。
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def save_single_model_results(
    model_name: str,
    labels: pd.Series,
    probabilities: np.ndarray,
    validation_ids: pd.Series,
    thresholds: np.ndarray,
    output_dir: Path,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """保存单个模型的指标、逐客户分数和固定阈值分析表。"""
    model_dir = output_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    # 汇总指标用于快速判断排序能力、少数类识别能力和概率质量。
    metrics = calculate_metrics(labels, probabilities)
    with (model_dir / "validation_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, ensure_ascii=False)

    # 固定阈值表回答“分数线为某个数时会发生什么”，但不适合跨模型直接判断优劣。
    threshold_table = make_threshold_table(labels, probabilities, thresholds)
    threshold_table.to_csv(model_dir / "threshold_analysis.csv", index=False)

    # 逐客户结果用于查找误判样本，也保证后续模型可以逐行公平比较。
    predictions = pd.DataFrame(
        {
            ID_COLUMN: validation_ids.to_numpy(),
            "TARGET": labels.to_numpy(),
            "predicted_risk": probabilities,
        }
    )
    predictions.to_csv(model_dir / "validation_predictions.csv", index=False)

    return metrics, threshold_table


def make_approval_rate_comparison(
    labels: pd.Series,
    model_probabilities: Dict[str, np.ndarray],
) -> pd.DataFrame:
    """在相同批准率下比较模型，这是比相同数值阈值更公平的策略比较。"""
    rows = []

    # 业务覆盖率从 10% 到 90%；例如 70% 表示只批准风险最低的 70% 客户。
    for target_approval_rate in np.arange(0.10, 1.00, 0.10):
        for model_name, probabilities in model_probabilities.items():
            # 风险越低越容易获批，因此用目标批准比例对应的风险分位数作为阈值。
            risk_threshold = float(np.quantile(probabilities, target_approval_rate))
            approved = probabilities < risk_threshold
            rejected = ~approved

            # 获批客群违约率衡量剩余组合风险；拒绝客群捕获率衡量风险拦截能力。
            approved_bad_rate = float(labels.to_numpy()[approved].mean())
            total_bad = int(labels.sum())
            captured_bad = int(labels.to_numpy()[rejected].sum())

            # 正常客户误拒率显示拦截风险客户时付出的客户机会成本。
            good_mask = labels.to_numpy() == 0
            rejected_good_rate = float((rejected & good_mask).sum() / good_mask.sum())
            rows.append(
                {
                    "model": model_name,
                    "target_approval_rate": float(target_approval_rate),
                    "actual_approval_rate": float(approved.mean()),
                    "risk_threshold": risk_threshold,
                    "bad_rate_among_approved": approved_bad_rate,
                    "bad_capture_rate": float(captured_bad / total_bad),
                    "good_customer_reject_rate": rejected_good_rate,
                }
            )

    return pd.DataFrame(rows)


def save_comparison_plots(
    labels: pd.Series,
    model_probabilities: Dict[str, np.ndarray],
    output_dir: Path,
) -> None:
    """把两个模型画在同一张 ROC 图和校准图中，便于直接观察差异。"""
    label_array = labels.to_numpy()
    display_names = {"balanced": "Balanced", "unbalanced": "Unbalanced"}

    # ROC 图只关心客户风险排序，不要求两个模型的概率刻度相同。
    figure, axis = plt.subplots(figsize=(7, 5))
    for model_name, probabilities in model_probabilities.items():
        false_positive_rate, true_positive_rate, _ = roc_curve(label_array, probabilities)
        axis.plot(false_positive_rate, true_positive_rate, label=display_names[model_name])
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random baseline")
    axis.set(
        title="Balanced vs unbalanced: ROC curve",
        xlabel="False positive rate",
        ylabel="True positive rate",
    )
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "comparison_roc_curve.png", dpi=160)
    plt.close(figure)

    # 校准图检验预测分数能否解释为真实违约概率，越接近 45 度线越好。
    figure, axis = plt.subplots(figsize=(7, 5))
    for model_name, probabilities in model_probabilities.items():
        observed_rate, predicted_rate = calibration_curve(
            label_array, probabilities, n_bins=10, strategy="quantile"
        )
        axis.plot(
            predicted_rate,
            observed_rate,
            marker="o",
            label=display_names[model_name],
        )
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    axis.set(
        title="Balanced vs unbalanced: calibration curve",
        xlabel="Mean predicted risk",
        ylabel="Observed default rate",
    )
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "comparison_calibration_curve.png", dpi=160)
    plt.close(figure)


def main() -> None:
    """用同一训练/验证集训练两版模型，保存单模型结果和公平对比结果。"""
    config = load_config()
    output_dir = config["output_dir"] / "class_weight_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # 读取主申请表并准备特征；这一部分与原始 baseline 完全一致。
    train, test = load_application_data(config["data_dir"])
    features, _, labels, numeric_columns, categorical_columns = make_features(train, test)

    # 两个模型只切分一次并共同使用结果，避免样本差异干扰 class_weight 对比。
    x_train, x_valid, y_train, y_valid = train_test_split(
        features,
        labels,
        test_size=config["validation_size"],
        random_state=config["random_state"],
        stratify=labels,
    )
    validation_ids = train.loc[y_valid.index, ID_COLUMN]
    logging.info("训练集 %s 条，验证集 %s 条", len(x_train), len(x_valid))

    # 继续沿用原 baseline 的阈值范围，保证单模型结果可以纵向核对。
    thresholds = np.arange(
        config["threshold_min"],
        config["threshold_max"] + config["threshold_step"] / 2,
        config["threshold_step"],
    )
    all_metrics = {}
    model_probabilities = {}

    # 循环中的唯一实验变量是 class_weight，其余模型参数、特征和样本都相同。
    for model_name, class_weight in MODEL_SETTINGS.items():
        logging.info("开始训练 %s 模型，class_weight=%s", model_name, class_weight)
        model = build_logistic_model(
            numeric_columns,
            categorical_columns,
            class_weight,
            config["random_state"],
        )
        model.fit(x_train, y_train)
        probabilities = model.predict_proba(x_valid)[:, 1]
        model_probabilities[model_name] = probabilities

        metrics, _ = save_single_model_results(
            model_name,
            y_valid,
            probabilities,
            validation_ids,
            thresholds,
            output_dir,
        )
        all_metrics[model_name] = metrics
        logging.info("%s 验证指标：%s", model_name, metrics)

    # 指标表一行一个模型，适合直接观察 AUC、KS 和 Brier 的相对变化。
    metrics_table = pd.DataFrame.from_dict(all_metrics, orient="index")
    metrics_table.index.name = "model"
    metrics_table.to_csv(output_dir / "metrics_comparison.csv")

    # 同批准率比较避免类别权重导致的分数尺度变化混淆策略表现。
    approval_table = make_approval_rate_comparison(y_valid, model_probabilities)
    approval_table.to_csv(output_dir / "approval_rate_comparison.csv", index=False)
    save_comparison_plots(y_valid, model_probabilities, output_dir)

    logging.info("对比结果已保存到 %s", output_dir)


if __name__ == "__main__":
    main()
