"""使用正式冠军模型的OOF预测，完成风险十分位和审批阈值策略分析。

OOF（Out-of-Fold）预测保证每名客户的风险概率都来自没有见过该客户的模型，
因此比训练集内预测更适合模拟真实审批决策。本脚本只分析策略，不重新训练模型。
"""

import json
from pathlib import Path
from typing import Dict, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from src.config import load_config


# 固定概率阈值用于回答“预测风险超过多少就拒绝”；审批率网格用于回答
# “如果业务希望批准固定比例客户，应把风险线放在哪里”。两种视角互为补充。
FIXED_RISK_THRESHOLDS = np.arange(0.02, 0.305, 0.01)
TARGET_APPROVAL_RATES = np.arange(0.50, 0.951, 0.05)


def calculate_strategy_point(
    labels: np.ndarray, probabilities: np.ndarray, threshold: float
) -> Dict[str, float]:
    """在一个风险阈值下计算审批、坏账拦截和正常客户误拒指标。"""
    # 风险低于阈值视为批准；达到或超过阈值视为拒绝或进入人工复核。
    approved = probabilities < threshold
    rejected = ~approved
    bad = labels == 1
    good = labels == 0

    approved_count = int(approved.sum())
    rejected_count = int(rejected.sum())
    approved_bad_count = int((approved & bad).sum())
    rejected_bad_count = int((rejected & bad).sum())
    approved_good_count = int((approved & good).sum())
    rejected_good_count = int((rejected & good).sum())

    # 坏客户捕获率以全部真实违约客户为分母；正常客户误拒率以全部正常客户为分母。
    # 二者必须一起看，否则“拒绝所有人”虽然捕获100%坏客户，却没有商业价值。
    return {
        "risk_threshold": float(threshold),
        "total_customers": int(len(labels)),
        "approved_count": approved_count,
        "rejected_or_review_count": rejected_count,
        "approval_rate": float(approved.mean()),
        "rejection_or_review_rate": float(rejected.mean()),
        "approved_bad_count": approved_bad_count,
        "approved_good_count": approved_good_count,
        "rejected_bad_count": rejected_bad_count,
        "rejected_good_count": rejected_good_count,
        "bad_rate_among_approved": float(approved_bad_count / approved_count) if approved_count else np.nan,
        "bad_capture_rate": float(rejected_bad_count / max(int(bad.sum()), 1)),
        "good_customer_reject_rate": float(rejected_good_count / max(int(good.sum()), 1)),
        # KS点等于当前阈值下的坏客户捕获率减去正常客户误拒率。
        "ks_at_threshold": float(
            rejected_bad_count / max(int(bad.sum()), 1)
            - rejected_good_count / max(int(good.sum()), 1)
        ),
        # 相对全体违约率下降多少，直观表达模型筛选对获批组合风险的改善。
        "approved_bad_rate_relative_reduction": float(
            1 - (approved_bad_count / approved_count) / labels.mean()
        ) if approved_count else np.nan,
    }


def make_decile_table(labels: np.ndarray, probabilities: np.ndarray) -> pd.DataFrame:
    """将客户按风险从低到高等分十组，并计算每组及累计风险表现。"""
    frame = pd.DataFrame({"actual_default": labels, "predicted_risk": probabilities})
    # risk_decile=1是风险最低10%，risk_decile=10是风险最高10%。
    frame["risk_decile"] = pd.qcut(
        frame["predicted_risk"], q=10, labels=False, duplicates="drop"
    ) + 1
    table = frame.groupby("risk_decile", observed=True).agg(
        customer_count=("actual_default", "size"),
        bad_count=("actual_default", "sum"),
        mean_predicted_risk=("predicted_risk", "mean"),
        actual_bad_rate=("actual_default", "mean"),
        minimum_predicted_risk=("predicted_risk", "min"),
        maximum_predicted_risk=("predicted_risk", "max"),
    ).reset_index()

    # Lift大于1表示该箱坏账率高于全体平均；最高风险箱Lift越高，风险越集中。
    portfolio_bad_rate = float(labels.mean())
    table["bad_rate_lift_vs_portfolio"] = table["actual_bad_rate"] / portfolio_bad_rate
    table["share_of_all_bad_customers"] = table["bad_count"] / table["bad_count"].sum()

    # 从最高风险箱向下累计，表示拒绝最高若干箱能拦截多少真实坏客户。
    reverse_bad = table["bad_count"].iloc[::-1].cumsum().iloc[::-1]
    reverse_customers = table["customer_count"].iloc[::-1].cumsum().iloc[::-1]
    table["bad_capture_if_reject_this_and_higher"] = reverse_bad / table["bad_count"].sum()
    table["rejection_rate_if_reject_this_and_higher"] = reverse_customers / table["customer_count"].sum()
    return table


def make_fixed_threshold_table(
    labels: np.ndarray, probabilities: np.ndarray, thresholds: Iterable[float]
) -> pd.DataFrame:
    """逐一评估业务人员容易理解的固定违约概率阈值。"""
    return pd.DataFrame(
        [calculate_strategy_point(labels, probabilities, threshold) for threshold in thresholds]
    )


def make_approval_target_table(
    labels: np.ndarray, probabilities: np.ndarray, approval_rates: Iterable[float]
) -> pd.DataFrame:
    """根据目标审批率反推风险阈值，并计算相应组合风险。"""
    rows = []
    for target_rate in approval_rates:
        # 风险概率的目标分位点就是理论审批线；使用真实比较后记录实际审批率。
        threshold = float(np.quantile(probabilities, target_rate))
        row = calculate_strategy_point(labels, probabilities, threshold)
        row["target_approval_rate"] = float(target_rate)
        rows.append(row)
    columns = ["target_approval_rate"] + [key for key in rows[0] if key != "target_approval_rate"]
    return pd.DataFrame(rows)[columns]


def find_ks_operating_point(labels: np.ndarray, probabilities: np.ndarray) -> Dict[str, float]:
    """找到样本内KS最大的风险阈值，作为统计参考而非自动业务最优点。"""
    false_positive_rate, true_positive_rate, thresholds = roc_curve(labels, probabilities)
    finite = np.isfinite(thresholds)
    differences = true_positive_rate[finite] - false_positive_rate[finite]
    best_position = int(np.argmax(differences))
    best_threshold = float(thresholds[finite][best_position])
    result = calculate_strategy_point(labels, probabilities, best_threshold)
    result["selection_note"] = "KS最大点只平衡区分效果，未考虑贷款收益、违约损失、人工审核成本和风险偏好。"
    return result


def make_three_way_policy(
    labels: np.ndarray,
    probabilities: np.ndarray,
    auto_approve_rate: float = 0.70,
    total_approve_rate: float = 0.85,
) -> pd.DataFrame:
    """示例展示自动批准、人工复核、拒绝三段式策略的客群规模与实际风险。"""
    approve_cutoff = float(np.quantile(probabilities, auto_approve_rate))
    reject_cutoff = float(np.quantile(probabilities, total_approve_rate))
    segment = np.where(
        probabilities < approve_cutoff,
        "1_自动批准",
        np.where(probabilities < reject_cutoff, "2_人工复核", "3_拒绝"),
    )
    frame = pd.DataFrame({"segment": segment, "actual_default": labels, "predicted_risk": probabilities})
    table = frame.groupby("segment", observed=True).agg(
        customer_count=("actual_default", "size"),
        actual_bad_count=("actual_default", "sum"),
        actual_bad_rate=("actual_default", "mean"),
        mean_predicted_risk=("predicted_risk", "mean"),
        minimum_predicted_risk=("predicted_risk", "min"),
        maximum_predicted_risk=("predicted_risk", "max"),
    ).reset_index()
    table["customer_share"] = table["customer_count"] / len(frame)
    table["share_of_all_bad_customers"] = table["actual_bad_count"] / labels.sum()
    table["policy_assumption"] = "示例：70%自动批准、15%人工复核、15%拒绝；不是利润最优策略"
    return table


def save_plots(
    deciles: pd.DataFrame,
    approval_targets: pd.DataFrame,
    three_way: pd.DataFrame,
    output_dir: Path,
) -> None:
    """保存风险分箱、审批权衡曲线和三段式策略图。"""
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    axes[0].bar(deciles["risk_decile"], deciles["actual_bad_rate"] * 100, color="#2474B5")
    axes[0].plot(deciles["risk_decile"], deciles["mean_predicted_risk"] * 100, marker="o", color="#E87524", label="Mean predicted risk")
    axes[0].set(title="Risk deciles: predicted vs actual", xlabel="Risk decile (1=lowest, 10=highest)", ylabel="Default rate (%)")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(deciles["risk_decile"], deciles["share_of_all_bad_customers"] * 100, color="#B84A62")
    axes[1].set(title="Bad-customer concentration by decile", xlabel="Risk decile", ylabel="Share of all bad customers (%)")
    axes[1].grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "risk_decile_analysis.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9, 6))
    axis.plot(approval_targets["approval_rate"] * 100, approval_targets["bad_rate_among_approved"] * 100, marker="o", label="Approved portfolio bad rate")
    axis.plot(approval_targets["approval_rate"] * 100, approval_targets["bad_capture_rate"] * 100, marker="o", label="Bad-customer capture rate")
    axis.plot(approval_targets["approval_rate"] * 100, approval_targets["good_customer_reject_rate"] * 100, marker="o", label="Good-customer reject rate")
    axis.set(title="Approval strategy trade-offs", xlabel="Approval rate (%)", ylabel="Rate (%)")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "approval_strategy_tradeoffs.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8.5, 5.5))
    # CSV保留中文业务名称；图片使用英文字体标签，避免不同电脑缺少中文字体时出现方框。
    plot_segments = ["Auto approve", "Manual review", "Reject"]
    axis.bar(plot_segments, three_way["customer_share"] * 100, color=["#36A269", "#D6A32A", "#B84A62"])
    for position, row in three_way.iterrows():
        axis.text(position, row["customer_share"] * 100 + 0.8, f"bad rate={row['actual_bad_rate']:.1%}", ha="center")
    axis.set(title="Illustrative three-way credit policy", xlabel="Decision segment", ylabel="Customer share (%)")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "three_way_policy_example.png", dpi=180)
    plt.close(figure)


def main() -> None:
    """读取正式冠军OOF预测并生成全部策略表、图和摘要。"""
    config = load_config()
    output_dir = config["output_dir"] / "credit_decision_strategy"
    output_dir.mkdir(parents=True, exist_ok=True)

    # targeted_data_governance文件同时包含原冠军与挑战者；明确选择raw_champion，
    # 因为上一轮正式决策是挑战者尚未满足稳定替换条件。
    prediction_path = config["output_dir"] / "targeted_data_governance" / "oof_predictions.csv"
    if not prediction_path.exists():
        raise FileNotFoundError(f"没有找到正式冠军OOF预测：{prediction_path}")
    predictions = pd.read_csv(prediction_path)
    labels = predictions["actual_default"].to_numpy(dtype="int8")
    probabilities = predictions["probability_raw_champion"].to_numpy(dtype="float64")

    if len(labels) != 307_511 or not np.isfinite(probabilities).all():
        raise ValueError("OOF预测数量或数值完整性异常，请先检查上一步五折输出。")

    deciles = make_decile_table(labels, probabilities)
    fixed = make_fixed_threshold_table(labels, probabilities, FIXED_RISK_THRESHOLDS)
    targets = make_approval_target_table(labels, probabilities, TARGET_APPROVAL_RATES)
    three_way = make_three_way_policy(labels, probabilities)
    ks_point = find_ks_operating_point(labels, probabilities)

    deciles.to_csv(output_dir / "risk_decile_table.csv", index=False)
    fixed.to_csv(output_dir / "fixed_risk_threshold_strategy.csv", index=False)
    targets.to_csv(output_dir / "target_approval_rate_strategy.csv", index=False)
    three_way.to_csv(output_dir / "three_way_policy_example.csv", index=False)
    with (output_dir / "ks_reference_operating_point.json").open("w", encoding="utf-8") as file:
        json.dump(ks_point, file, ensure_ascii=False, indent=2)

    # 摘要记录数据口径和关键验证结论，保证未来查看时知道数字来自哪版模型。
    monotonic = bool(deciles["actual_bad_rate"].is_monotonic_increasing)
    summary = {
        "model": "formal raw champion: XGBoost + Application + Bureau + Previous + Installments",
        "prediction_type": "5-fold out-of-fold probabilities",
        "customer_count": int(len(labels)),
        "portfolio_bad_rate": float(labels.mean()),
        "decile_bad_rate_strictly_monotonic_non_decreasing": monotonic,
        "lowest_decile_bad_rate": float(deciles.iloc[0]["actual_bad_rate"]),
        "highest_decile_bad_rate": float(deciles.iloc[-1]["actual_bad_rate"]),
        "highest_decile_lift": float(deciles.iloc[-1]["bad_rate_lift_vs_portfolio"]),
        "highest_20pct_bad_capture_rate": float(deciles.loc[deciles["risk_decile"] >= 9, "bad_count"].sum() / labels.sum()),
        "important_limit": "未提供贷款收益、LGD、EAD、资金成本和人工审核成本，因此不能宣称某个阈值利润最优。",
    }
    with (output_dir / "decision_strategy_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    save_plots(deciles, targets, three_way, output_dir)


if __name__ == "__main__":
    main()
