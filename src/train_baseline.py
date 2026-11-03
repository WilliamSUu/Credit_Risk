"""训练主申请表逻辑回归基线，并生成验证结果、阈值表和诊断图。"""

import json
import logging

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.config import load_config
from src.data import ID_COLUMN, TARGET_COLUMN, load_application_data
from src.evaluation import calculate_metrics, make_threshold_table, save_diagnostic_plots
from src.features import build_preprocessor, make_features


def main() -> None:
    """按照“读配置 -> 读数据 -> 切分 -> 训练 -> 评估 -> 保存结果”的顺序执行。"""
    # 读取数据目录、输出目录、验证比例和随机种子等可调整参数。
    config = load_config()
    output_dir = config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # 第一阶段只读取当前申请主表；等基线确认可靠后再增加历史行为聚合特征。
    train, test = load_application_data(config["data_dir"])
    # 拆出标签并识别特征类型；测试表暂不参与训练或验证。
    features, _, labels, numeric_columns, categorical_columns = make_features(train, test)
    logging.info("Loaded %s applications and %s features", len(train), features.shape[1])
    logging.info("Training payment-difficulty rate: %.4f", labels.mean())

    # 分层抽样让训练集和验证集的违约比例大致一致。
    # 验证集完全留出，用来模拟模型遇到未见客户时的表现。
    x_train, x_valid, y_train, y_valid = train_test_split(
        features,
        labels,
        test_size=config["validation_size"],
        random_state=config["random_state"],
        stratify=labels,
    )

    # 不加类别权重的逻辑回归现在作为正式 baseline。
    # 前序实验表明 balanced 与 unbalanced 的排序能力相近，但 unbalanced 的概率校准明显更好。
    model = Pipeline(
        steps=[
            # 预处理器只在训练部分拟合，防止用验证数据计算中位数/均值等信息。
            ("preprocessor", build_preprocessor(numeric_columns, categorical_columns)),
            (
                "classifier",
                LogisticRegression(
                    # 增加迭代上限，让优化器有机会收敛。
                    max_iter=500,
                    # None 表示每条训练记录权重相同，保留原始违约率信息。
                    class_weight=None,
                    # liblinear 适用于这种中等规模稀疏特征的二分类逻辑回归。
                    solver="liblinear",
                    random_state=config["random_state"],
                ),
            ),
        ]
    )
    # fit 会在训练数据上学习填补值、缩放参数、类别编码和逻辑回归系数。
    model.fit(x_train, y_train)

    # predict_proba 的第 2 列是预测为违约（TARGET=1）的分数/概率。
    validation_probabilities = model.predict_proba(x_valid)[:, 1]
    # 计算整体排序、违约类识别、KS 和概率误差指标。
    metrics = calculate_metrics(y_valid, validation_probabilities)
    logging.info("Validation metrics: %s", metrics)

    # Logistic Regression 的默认分类阈值不一定适合授信；遍历阈值观察覆盖率与风险变化。
    thresholds = np.arange(
        config["threshold_min"],
        config["threshold_max"] + config["threshold_step"] / 2,
        config["threshold_step"],
    )
    threshold_table = make_threshold_table(y_valid, validation_probabilities, thresholds)
    # 保存 CSV 便于在 Excel 中筛选、画业务曲线或比较后续模型。
    threshold_table.to_csv(output_dir / "threshold_analysis.csv", index=False)
    # 输出两张用于模型诊断的图，而不只依赖一个汇总分数。
    save_diagnostic_plots(y_valid, validation_probabilities, output_dir)

    # 保存验证客户 ID、真实标签和预测分数，便于逐样本检查误判。
    # ID 只用于追踪行，不进入模型特征。
    prediction_output = train.loc[y_valid.index, [ID_COLUMN]].copy()
    prediction_output["TARGET"] = y_valid
    prediction_output["predicted_risk"] = validation_probabilities
    prediction_output.to_csv(output_dir / "validation_predictions.csv", index=False)

    # 保存机器可读指标，后续可与 LightGBM、历史表增强版等结果并排比较。
    with (output_dir / "validation_metrics.json").open("w", encoding="utf-8") as result_file:
        json.dump(metrics, result_file, indent=2)

    logging.info("Saved baseline outputs to %s", output_dir)
    logging.info("Modeled target column: %s", TARGET_COLUMN)


if __name__ == "__main__":
    main()
