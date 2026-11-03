"""计算验证指标、不同风险阈值下的业务权衡，并保存诊断图。"""

from pathlib import Path
from typing import Dict, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)


def calculate_ks(labels: Iterable[int], probabilities: Iterable[float]) -> float:
    """计算 KS：违约客户累计捕获率与非违约客户误报率的最大差值。

    KS 越大，模型区分好坏客户的能力通常越强；它不代表概率已经校准。
    """
    # roc_curve 返回多个阈值下的 FPR（好客户被错判高风险比例）和 TPR（违约客户捕获率）。
    false_positive_rate, true_positive_rate, _ = roc_curve(labels, probabilities)
    # KS 就是所有阈值中 TPR - FPR 的最大值。
    return float(np.max(true_positive_rate - false_positive_rate))


def calculate_metrics(
    labels: Iterable[int], probabilities: Iterable[float]
) -> Dict[str, float]:
    """在未参与训练的验证样本上计算排序能力和概率误差指标。"""
    # ROC-AUC 衡量随机抽取一名违约者和一名正常客户时，前者风险分更高的概率。
    # PR-AUC 综合精确率与召回率；违约样本少时，它比单看 ROC-AUC 更能反映正类识别质量。
    return {
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "pr_auc": float(average_precision_score(labels, probabilities)),
        "ks": calculate_ks(labels, probabilities),
        # Brier 是预测概率与 0/1 结果的均方误差，越低越好，兼顾校准和区分能力。
        "brier_score": float(brier_score_loss(labels, probabilities)),
        # 验证集真实违约比例；它是 PR-AUC 的重要参考基线。
        "default_rate": float(np.mean(labels)),
    }


def make_threshold_table(
    labels: Iterable[int], probabilities: Iterable[float], thresholds: Iterable[float]
) -> pd.DataFrame:
    """对一组风险分阈值计算审批覆盖率和风险权衡。

    这里没有编造每笔贷款的利润/损失金额，所以输出的是运营指标，不是利润最优点。
    """
    # 转成 NumPy 数组，便于按阈值对所有验证客户做向量化筛选。
    labels_array = np.asarray(labels)
    probability_array = np.asarray(probabilities)
    rows = []

    # 风险分达到或超过阈值时，假设进入拒绝/人工复核；低于阈值的视为批准。
    for threshold in thresholds:
        predicted_bad = probability_array >= threshold
        # 混淆矩阵按 [真实正常, 真实违约] 顺序展开为 TN、FP、FN、TP。
        true_negative, false_positive, false_negative, true_positive = (
            confusion_matrix(labels_array, predicted_bad.astype(int), labels=[0, 1])
            .ravel()
        )
        approved = ~predicted_bad
        approved_count = int(approved.sum())
        # 批准客户中真实违约的比例；如果无人获批则没有定义，用 NaN 表示。
        approved_bad_rate = (
            float(labels_array[approved].mean()) if approved_count else np.nan
        )
        rows.append(
            {
                # 当前使用的高风险判定线。
                "risk_threshold": float(threshold),
                # 所有验证客户里，被判定为低风险并获批的比例。
                "approval_rate": float(approved.mean()),
                # 获批客群中实际违约者占比。
                "bad_rate_among_approved": approved_bad_rate,
                # 所有真实违约客户中被拦截/复核的比例（TPR）。
                "bad_capture_rate": float(true_positive / max(true_positive + false_negative, 1)),
                # 所有正常客户中被误拦截的比例（FPR）。
                "good_customer_reject_rate": float(false_positive / max(false_positive + true_negative, 1)),
            }
        )

    return pd.DataFrame(rows)


def save_diagnostic_plots(
    labels: Iterable[int], probabilities: Iterable[float], output_dir: Path
) -> None:
    """将 ROC 曲线和概率校准曲线保存为 PNG 图片。"""
    # 确保输出目录存在；即使第一次运行也不要求手动创建 outputs 文件夹。
    output_dir.mkdir(parents=True, exist_ok=True)
    labels_array = np.asarray(labels)
    probability_array = np.asarray(probabilities)

    # ROC 横轴是假阳性率，纵轴是真阳性率；曲线越靠左上越好。
    false_positive_rate, true_positive_rate, _ = roc_curve(
        labels_array, probability_array
    )
    fig, axis = plt.subplots(figsize=(7, 5))
    axis.plot(false_positive_rate, true_positive_rate, label="Logistic Regression")
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random baseline")
    # 对角虚线表示随机排序；曲线下面积即 ROC-AUC。
    axis.set(title="Validation ROC curve", xlabel="False positive rate", ylabel="True positive rate")
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "roc_curve.png", dpi=160)
    plt.close(fig)

    # 按预测风险分位数分箱，比较每箱平均预测概率与真实违约比例。
    observed_rate, predicted_rate = calibration_curve(
        labels_array, probability_array, n_bins=10, strategy="quantile"
    )
    fig, axis = plt.subplots(figsize=(7, 5))
    axis.plot(predicted_rate, observed_rate, marker="o", label="Logistic Regression")
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    # 曲线越接近 45 度虚线，概率数值越接近实际发生频率。
    axis.set(title="Validation calibration curve", xlabel="Mean predicted risk", ylabel="Observed default rate")
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "calibration_curve.png", dpi=160)
    plt.close(fig)
