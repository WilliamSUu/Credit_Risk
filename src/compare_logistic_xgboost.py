"""在同一份验证数据上比较正式 Logistic baseline 与 XGBoost。"""

import json
import logging
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.model_selection import train_test_split

from src.config import load_config
from src.data import ID_COLUMN, load_application_data
from src.evaluation import calculate_metrics, make_threshold_table
from src.features import build_preprocessor, make_features


# 内部名称用于文件夹和表格；显示名称用于图片图例。
MODEL_NAMES = {
    "logistic_baseline": "Logistic Regression",
    "xgboost": "XGBoost",
}


def build_xgboost(random_state: int) -> xgb.XGBClassifier:
    """建立首版 XGBoost，并通过 early stopping 控制过拟合。"""
    # XGBoost 会依次增加决策树，每棵新树用于修正此前模型仍然犯的错误。
    # 与逻辑回归相比，它能自然学习非线性关系和多个变量之间的交互。
    return xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        # 2000 是允许的最大树数量，并不代表最终一定使用 2000 棵树。
        n_estimators=2000,
        # 较低学习率让每棵树只做小幅修正，通常更加稳健。
        learning_rate=0.03,
        # 树深控制单棵树复杂度；过深容易记住训练样本噪声。
        max_depth=6,
        # 一个叶节点需要积累足够样本权重，减少只服务少量样本的分裂。
        min_child_weight=20,
        # 每棵树随机使用 80% 训练样本和 80% 特征，降低树之间的相关性。
        subsample=0.8,
        colsample_bytree=0.8,
        # L1 和 L2 正则化抑制过于激进的叶节点分数。
        reg_alpha=0.1,
        reg_lambda=1.0,
        # hist 是 XGBoost 的高效直方图建树算法，适合当前表格数据规模。
        tree_method="hist",
        # 验证 AUC 连续 100 轮未改善时停止，并保留最佳轮次。
        early_stopping_rounds=100,
        random_state=random_state,
        n_jobs=-1,
    )


def save_model_outputs(
    model_name: str,
    labels: pd.Series,
    probabilities: np.ndarray,
    validation_ids: pd.Series,
    thresholds: np.ndarray,
    output_dir: Path,
) -> Dict[str, float]:
    """保存单个模型的验证指标、阈值分析和逐客户预测。"""
    model_dir = output_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    # 指标同时覆盖排序能力、少数类识别、KS 区分度和概率误差。
    metrics = calculate_metrics(labels, probabilities)
    with (model_dir / "validation_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, ensure_ascii=False)

    # 固定阈值表用于理解单模型的分数含义；跨模型比较优先看同批准率表。
    make_threshold_table(labels, probabilities, thresholds).to_csv(
        model_dir / "threshold_analysis.csv", index=False
    )

    # 保存验证样本的 ID、真实标签和分数，便于定位两个模型判断不同的客户。
    predictions = pd.DataFrame(
        {
            ID_COLUMN: validation_ids.to_numpy(),
            "TARGET": labels.to_numpy(),
            "predicted_risk": probabilities,
        }
    )
    predictions.to_csv(model_dir / "validation_predictions.csv", index=False)
    return metrics


def make_approval_rate_comparison(
    labels: pd.Series, model_probabilities: Dict[str, np.ndarray]
) -> pd.DataFrame:
    """在相同批准率下比较两个模型的实际风险筛选效果。"""
    label_array = labels.to_numpy()
    rows = []

    # 从批准最低风险 10% 到 90%，覆盖严格与宽松审批场景。
    for target_approval_rate in np.arange(0.10, 1.00, 0.10):
        for model_name, probabilities in model_probabilities.items():
            # 两个模型的分数尺度可能不同，所以通过分位数得到相同批准比例。
            risk_threshold = float(np.quantile(probabilities, target_approval_rate))
            approved = probabilities < risk_threshold
            rejected = ~approved

            # 相同批准率下，获批坏账率越低，说明模型保留的客户组合越好。
            approved_bad_rate = float(label_array[approved].mean())
            captured_bad_rate = float(label_array[rejected].sum() / label_array.sum())

            # 正常客户误拒率衡量风险控制可能付出的正常客户机会成本。
            good_mask = label_array == 0
            rejected_good_rate = float((rejected & good_mask).sum() / good_mask.sum())
            rows.append(
                {
                    "model": model_name,
                    "target_approval_rate": float(target_approval_rate),
                    "actual_approval_rate": float(approved.mean()),
                    "risk_threshold": risk_threshold,
                    "bad_rate_among_approved": approved_bad_rate,
                    "bad_capture_rate": captured_bad_rate,
                    "good_customer_reject_rate": rejected_good_rate,
                }
            )

    return pd.DataFrame(rows)


def save_comparison_plots(
    labels: pd.Series,
    model_probabilities: Dict[str, np.ndarray],
    output_dir: Path,
) -> None:
    """保存 ROC、Precision–Recall 和概率校准三张对比图。"""
    label_array = labels.to_numpy()

    # ROC 曲线展示不同正常客户误拒水平下，可以识别多少违约客户。
    figure, axis = plt.subplots(figsize=(7, 5))
    for model_name, probabilities in model_probabilities.items():
        false_positive_rate, true_positive_rate, _ = roc_curve(label_array, probabilities)
        axis.plot(false_positive_rate, true_positive_rate, label=MODEL_NAMES[model_name])
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random baseline")
    axis.set(title="Logistic vs XGBoost: ROC curve", xlabel="False positive rate", ylabel="True positive rate")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "comparison_roc_curve.png", dpi=160)
    plt.close(figure)

    # PR 曲线更关注少数违约类；随机水平大约等于验证集真实违约率。
    figure, axis = plt.subplots(figsize=(7, 5))
    for model_name, probabilities in model_probabilities.items():
        precision, recall, _ = precision_recall_curve(label_array, probabilities)
        axis.plot(recall, precision, label=MODEL_NAMES[model_name])
    axis.axhline(label_array.mean(), linestyle="--", color="gray", label="Random baseline")
    axis.set(title="Logistic vs XGBoost: Precision-Recall curve", xlabel="Recall", ylabel="Precision")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "comparison_pr_curve.png", dpi=160)
    plt.close(figure)

    # 校准图检查模型给出的概率是否接近相应客户组的真实违约频率。
    figure, axis = plt.subplots(figsize=(7, 5))
    for model_name, probabilities in model_probabilities.items():
        observed_rate, predicted_rate = calibration_curve(
            label_array, probabilities, n_bins=10, strategy="quantile"
        )
        axis.plot(predicted_rate, observed_rate, marker="o", label=MODEL_NAMES[model_name])
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    axis.set(title="Logistic vs XGBoost: calibration curve", xlabel="Mean predicted risk", ylabel="Observed default rate")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "comparison_calibration_curve.png", dpi=160)
    plt.close(figure)


def save_xgboost_importance(
    model: xgb.XGBClassifier, feature_names: np.ndarray, output_dir: Path
) -> None:
    """保存 XGBoost 的 Gain 特征重要性和前 20 名图表。"""
    # Gain 表示某个特征用于树分裂时平均带来的目标函数改善。
    raw_importance = model.get_booster().get_score(importance_type="gain")
    importance = pd.DataFrame({"feature": feature_names})

    # XGBoost 对稀疏矩阵的特征可能使用 f0、f1 等位置名称，需要映射回真实列名。
    position_to_gain = {
        int(key[1:]): value for key, value in raw_importance.items() if key.startswith("f")
    }
    importance["gain_importance"] = [
        position_to_gain.get(position, 0.0) for position in range(len(feature_names))
    ]
    importance = importance.sort_values("gain_importance", ascending=False)

    # gain_share 表示每个字段在全部分裂增益中所占比例。
    total_gain = importance["gain_importance"].sum()
    importance["gain_share"] = importance["gain_importance"] / total_gain
    importance.to_csv(output_dir / "xgboost_feature_importance.csv", index=False)

    # 图片只展示前 20 个字段，完整排名保留在 CSV 中。
    top_features = importance.head(20).sort_values("gain_importance")
    figure, axis = plt.subplots(figsize=(9, 7))
    axis.barh(top_features["feature"], top_features["gain_importance"])
    axis.set(title="XGBoost top 20 feature importance", xlabel="Average gain")
    figure.tight_layout()
    figure.savefig(output_dir / "xgboost_feature_importance.png", dpi=160)
    plt.close(figure)


def main() -> None:
    """训练正式 baseline 和 XGBoost，并在相同验证样本上生成完整比较。"""
    config = load_config()
    output_dir = config["output_dir"] / "logistic_vs_xgboost"
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # 本轮仍只使用 application 主表，确保指标变化来自模型，而不是新增历史数据。
    train, test = load_application_data(config["data_dir"])
    features, _, labels, numeric_columns, categorical_columns = make_features(train, test)

    # 只切分一次，Logistic 与 XGBoost 共用相同训练和验证客户。
    x_train, x_valid, y_train, y_valid = train_test_split(
        features,
        labels,
        test_size=config["validation_size"],
        random_state=config["random_state"],
        stratify=labels,
    )
    validation_ids = train.loc[y_valid.index, ID_COLUMN]
    logging.info("训练集 %s 条，验证集 %s 条", len(x_train), len(x_valid))

    # 预处理器只在训练集拟合，然后用同一规则转换训练集和验证集。
    # 两个分类器共享相同的模型输入，从而把差异集中在模型结构本身。
    preprocessor = build_preprocessor(numeric_columns, categorical_columns)
    transformed_train = preprocessor.fit_transform(x_train)
    transformed_valid = preprocessor.transform(x_valid)

    # 正式 baseline 是不加类别权重的逻辑回归。
    logistic_model = LogisticRegression(
        max_iter=500,
        class_weight=None,
        solver="liblinear",
        random_state=config["random_state"],
    )
    logistic_model.fit(transformed_train, y_train)
    logistic_probabilities = logistic_model.predict_proba(transformed_valid)[:, 1]

    # XGBoost 使用相同输入，并监控同一验证集的 AUC 做 early stopping。
    xgboost_model = build_xgboost(config["random_state"])
    xgboost_model.fit(
        transformed_train,
        y_train,
        eval_set=[(transformed_valid, y_valid)],
        verbose=100,
    )
    xgboost_probabilities = xgboost_model.predict_proba(transformed_valid)[:, 1]

    model_probabilities = {
        "logistic_baseline": logistic_probabilities,
        "xgboost": xgboost_probabilities,
    }
    thresholds = np.arange(
        config["threshold_min"],
        config["threshold_max"] + config["threshold_step"] / 2,
        config["threshold_step"],
    )

    # 以相同格式保存两个模型的结果，便于逐项核对。
    metrics = {}
    for model_name, probabilities in model_probabilities.items():
        metrics[model_name] = save_model_outputs(
            model_name,
            y_valid,
            probabilities,
            validation_ids,
            thresholds,
            output_dir,
        )
        logging.info("%s 验证指标：%s", model_name, metrics[model_name])

    # best_iteration 是 early stopping 选出的最佳树轮次。
    metrics_table = pd.DataFrame.from_dict(metrics, orient="index")
    metrics_table.index.name = "model"
    metrics_table["best_iteration"] = np.nan
    metrics_table.loc["xgboost", "best_iteration"] = xgboost_model.best_iteration
    metrics_table.to_csv(output_dir / "metrics_comparison.csv")

    approval_table = make_approval_rate_comparison(y_valid, model_probabilities)
    approval_table.to_csv(output_dir / "approval_rate_comparison.csv", index=False)
    save_comparison_plots(y_valid, model_probabilities, output_dir)

    # 从已经拟合的预处理器取得独热编码后的真实特征名，再生成重要性排名。
    feature_names = preprocessor.get_feature_names_out()
    save_xgboost_importance(xgboost_model, feature_names, output_dir)
    logging.info("Logistic 与 XGBoost 对比结果已保存到 %s", output_dir)


if __name__ == "__main__":
    main()
