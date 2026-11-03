"""对当前信贷冠军模型执行数据质量诊断、异常治理和概率校准评估。

本脚本回答三个风控建模问题：原始聚合特征是否存在异常极值；训练集与
Kaggle测试集的数据完整性是否稳定；治理异常值后，模型在严格五折验证中
是否仍保持或改善区分能力与概率质量。
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from scipy.optimize import minimize
from scipy.special import expit

from src.bureau_features import CUSTOMER_ID, aggregate_bureau, load_bureau
from src.config import load_config
from src.cross_validate_champion import build_cv_xgboost
from src.data import load_application_data
from src.evaluation import calculate_metrics
from src.features import build_preprocessor, make_features
from src.installment_features import build_installment_features
from src.previous_application_features import build_previous_application_features


# 只治理有明确业务含义的“比率”字段，不对全部金额和时间字段盲目截尾。
# 0.5%与99.5%兼顾稳健性和信息保留；阈值会在每一折训练集内单独学习。
LOWER_QUANTILE = 0.005
UPPER_QUANTILE = 0.995
METRICS = ["roc_auc", "pr_auc", "ks", "brier_score"]
EXPERIMENT_NAMES = {
    "raw_champion": "当前冠军：原始特征",
    "governed_champion": "候选模型：比率异常治理",
}
PLOT_NAMES = {
    "raw_champion": "Raw champion",
    "governed_champion": "Ratio-governed candidate",
}


def identify_source(column: str) -> str:
    """按照字段名前缀，把字段归入主申请或三类历史数据来源。"""
    if column.startswith("BUREAU_") or column.startswith("HAS_BUREAU"):
        return "Bureau外部征信"
    if column.startswith("PREV_") or column.startswith("HAS_PREVIOUS"):
        return "Previous历史申请"
    if column.startswith("INST_") or column.startswith("HAS_INSTALLMENT"):
        return "Installments还款"
    return "Application主申请"


def build_complete_tables(config: Dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """构造训练、测试冠军特征表，并保留标签和客户编号用于审计。"""
    application_train, application_test = load_application_data(config["data_dir"])
    # 三张历史表都先聚合到“一名客户一行”，再与主申请表连接。
    # 这样不会因一名客户有多笔历史贷款而把主表客户重复多次。
    bureau = aggregate_bureau(load_bureau(config["history_data_dir"]))
    previous = build_previous_application_features(config["history_data_dir"])
    installments = build_installment_features(config["history_data_dir"])
    history = bureau.merge(previous, on=CUSTOMER_ID, how="outer", validate="one_to_one")
    history = history.merge(installments, on=CUSTOMER_ID, how="outer", validate="one_to_one")
    training = application_train.merge(history, on=CUSTOMER_ID, how="left", validate="one_to_one")
    testing = application_test.merge(history, on=CUSTOMER_ID, how="left", validate="one_to_one")

    # HAS字段表示客户是否存在对应历史；无历史记录应明确记为0，而不是普通缺失。
    for table in (training, testing):
        for column in table.columns:
            if column.startswith("HAS_"):
                table[column] = table[column].fillna(0).astype("int8")

    customer_ids = training[CUSTOMER_ID].copy()
    test_ids = testing[CUSTOMER_ID].copy()
    del test_ids
    # make_features统一删除标签与客户编号，并执行与正式冠军完全相同的字段对齐和哨兵值处理。
    train_features, test_features, labels, _, _ = make_features(training, testing)
    return train_features, test_features, labels, customer_ids


def save_missingness_diagnostics(train: pd.DataFrame, test: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """比较训练和测试缺失率，识别上线前可能出现的数据完整性漂移。"""
    rows = []
    for column in train.columns:
        train_missing = float(train[column].isna().mean())
        test_missing = float(test[column].isna().mean())
        rows.append({
            "feature": column,
            "source": identify_source(column),
            "train_missing_rate": train_missing,
            "test_missing_rate": test_missing,
            "absolute_gap": abs(test_missing - train_missing),
            "signed_test_minus_train": test_missing - train_missing,
            # 历史表整体缺失经常表示客户没有该类历史，并不自动等同于数据错误。
            "missingness_type_note": "可能是结构性无历史" if identify_source(column) != "Application主申请" else "主申请采集缺失",
        })
    result = pd.DataFrame(rows).sort_values("absolute_gap", ascending=False)
    result.to_csv(output_dir / "missingness_train_test_all_features.csv", index=False)
    result.head(50).to_csv(output_dir / "missingness_drift_top50.csv", index=False)

    # 按来源汇总可快速判断漂移集中在哪张业务表，而不是只看数百个字段。
    source_summary = result.groupby("source", as_index=False).agg(
        feature_count=("feature", "size"),
        mean_train_missing_rate=("train_missing_rate", "mean"),
        mean_test_missing_rate=("test_missing_rate", "mean"),
        mean_absolute_gap=("absolute_gap", "mean"),
        max_absolute_gap=("absolute_gap", "max"),
    )
    source_summary.to_csv(output_dir / "missingness_summary_by_source.csv", index=False)

    top = result.head(20).sort_values("absolute_gap")
    figure, axis = plt.subplots(figsize=(10, 8))
    axis.barh(top["feature"], top["absolute_gap"] * 100, color="#D8792B")
    axis.set(title="Top 20 train-test missing-rate gaps", xlabel="Absolute missing-rate gap (percentage points)")
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "missingness_drift_top20.png", dpi=180)
    plt.close(figure)
    return result


def save_distribution_diagnostics(train: pd.DataFrame, output_dir: Path) -> Tuple[pd.DataFrame, List[str]]:
    """输出比率字段的分位数、非有限值、负值和极端值数量。"""
    # 使用“_RATIO”而不是宽泛的“RATIO”，避免把DURATION（期限）误认成比率。
    ratio_columns = [column for column in train.columns if "_RATIO" in column and pd.api.types.is_numeric_dtype(train[column])]
    rows = []
    quantiles = [0.0, 0.01, 0.05, 0.5, 0.95, 0.99, 0.995, 0.999, 1.0]
    for column in ratio_columns:
        values = pd.to_numeric(train[column], errors="coerce")
        finite = values.replace([np.inf, -np.inf], np.nan).dropna()
        points = finite.quantile(quantiles) if len(finite) else pd.Series(index=quantiles, dtype=float)
        rows.append({
            "feature": column, "source": identify_source(column),
            "non_missing_count": int(values.notna().sum()),
            "non_finite_count": int(np.isinf(values.to_numpy(dtype=float, na_value=np.nan)).sum()),
            "negative_count": int((finite < 0).sum()),
            "zero_count": int((finite == 0).sum()),
            "p0": points.get(0.0), "p1": points.get(0.01), "p5": points.get(0.05),
            "p50": points.get(0.5), "p95": points.get(0.95), "p99": points.get(0.99),
            "p99_5": points.get(0.995), "p99_9": points.get(0.999), "max": points.get(1.0),
            "above_p99_5_count": int((finite > points.get(0.995, np.inf)).sum()) if len(finite) else 0,
        })
    result = pd.DataFrame(rows).sort_values("max", ascending=False)
    result.to_csv(output_dir / "ratio_distribution_quantiles.csv", index=False)

    # 重点画出最大值相对P99.5最夸张的字段，展示少数极端点如何拉长坐标轴。
    plot_table = result.replace([np.inf, -np.inf], np.nan).dropna(subset=["p99_5", "max"]).copy()
    plot_table["max_to_p99_5"] = plot_table["max"] / plot_table["p99_5"].replace(0, np.nan)
    selected = plot_table.sort_values("max_to_p99_5", ascending=False).head(12).sort_values("max_to_p99_5")
    figure, axis = plt.subplots(figsize=(10, 7))
    axis.barh(selected["feature"], selected["max_to_p99_5"], color="#B84A62")
    axis.set(title="Ratio-feature tail severity", xlabel="Maximum divided by P99.5 (larger means heavier extreme tail)")
    axis.axvline(1, color="black", linewidth=1)
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "ratio_extreme_tail_severity.png", dpi=180)
    plt.close(figure)
    return result, ratio_columns


def learn_and_apply_governance(x_train: pd.DataFrame, x_valid: pd.DataFrame, ratio_columns: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """只用本折训练数据学习截尾线，并给训练/验证数据增加异常标记。"""
    governed_train = x_train.copy()
    governed_valid = x_valid.copy()
    audit_rows = []
    for column in ratio_columns:
        # 正负无穷通常来自异常分母，先转为缺失，使后续中位数填补可以安全处理。
        training_values = pd.to_numeric(governed_train[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        valid_values = pd.to_numeric(governed_valid[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        lower = float(training_values.quantile(LOWER_QUANTILE))
        upper = float(training_values.quantile(UPPER_QUANTILE))

        # 标记告诉模型“该值原本越界”，同时截尾限制异常数值本身的杠杆影响。
        flag_name = f"{column}_OUTLIER_FLAG"
        governed_train[flag_name] = ((training_values < lower) | (training_values > upper)).astype("int8")
        governed_valid[flag_name] = ((valid_values < lower) | (valid_values > upper)).astype("int8")
        governed_train[column] = training_values.clip(lower=lower, upper=upper)
        governed_valid[column] = valid_values.clip(lower=lower, upper=upper)
        audit_rows.append({
            "feature": column, "lower_bound": lower, "upper_bound": upper,
            "train_outlier_count": int(governed_train[flag_name].sum()),
            "valid_outlier_count": int(governed_valid[flag_name].sum()),
        })
    return governed_train, governed_valid, pd.DataFrame(audit_rows)


def expected_calibration_error(labels: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    """计算等频分箱ECE；它是预测风险与真实违约率差距的加权平均。"""
    frame = pd.DataFrame({"label": labels, "probability": probabilities})
    frame["bin"] = pd.qcut(frame["probability"], q=bins, labels=False, duplicates="drop")
    grouped = frame.groupby("bin", observed=True).agg(count=("label", "size"), actual=("label", "mean"), predicted=("probability", "mean"))
    return float(((grouped["count"] / len(frame)) * (grouped["actual"] - grouped["predicted"]).abs()).sum())


def make_calibration_table(labels: np.ndarray, probabilities: np.ndarray, experiment: str) -> pd.DataFrame:
    """按预测风险十分位比较平均预测风险和实际违约率。"""
    frame = pd.DataFrame({"actual_default": labels, "predicted_risk": probabilities})
    frame["risk_decile"] = pd.qcut(frame["predicted_risk"], 10, labels=False, duplicates="drop") + 1
    table = frame.groupby("risk_decile", observed=True).agg(
        customer_count=("actual_default", "size"),
        mean_predicted_risk=("predicted_risk", "mean"),
        actual_default_rate=("actual_default", "mean"),
        minimum_predicted_risk=("predicted_risk", "min"),
        maximum_predicted_risk=("predicted_risk", "max"),
    ).reset_index()
    table.insert(0, "experiment", experiment)
    table["calibration_gap"] = table["mean_predicted_risk"] - table["actual_default_rate"]
    return table


def estimate_calibration_intercept_slope(labels: np.ndarray, probabilities: np.ndarray) -> Tuple[float, float]:
    """用稳定的二项对数似然估计校准截距和斜率，避免极端概率造成数值警告。"""
    clipped = np.clip(probabilities.astype("float64"), 1e-6, 1 - 1e-6)
    log_odds = np.log(clipped / (1 - clipped))
    targets = labels.astype("float64")

    # 参数0是截距、参数1是斜率；logaddexp稳定计算log(1 + exp(x))。
    def objective(parameters: np.ndarray) -> Tuple[float, np.ndarray]:
        linear = parameters[0] + parameters[1] * log_odds
        loss = np.logaddexp(0, linear).sum() - np.dot(targets, linear)
        residual = expit(linear) - targets
        gradient = np.array([residual.sum(), np.dot(residual, log_odds)])
        return float(loss), gradient

    result = minimize(objective, x0=np.array([0.0, 1.0]), jac=True, method="BFGS")
    if not result.success and not np.isfinite(result.x).all():
        raise RuntimeError(f"校准截距/斜率估计失败：{result.message}")
    return float(result.x[0]), float(result.x[1])


def run_paired_cv(features: pd.DataFrame, labels: pd.Series, ratio_columns: List[str], config: Dict, output_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """在完全相同的五折客户上比较原始冠军与治理候选，并保存OOF预测。"""
    splitter = StratifiedKFold(n_splits=config["cross_validation_folds"], shuffle=True, random_state=config["random_state"])
    folds = list(splitter.split(features, labels))
    oof = {name: np.zeros(len(features), dtype="float32") for name in EXPERIMENT_NAMES}
    fold_rows, audit_tables = [], []

    for fold_number, (train_positions, valid_positions) in enumerate(folds, start=1):
        logging.info("开始第%s/%s折", fold_number, len(folds))
        raw_train, raw_valid = features.iloc[train_positions].copy(), features.iloc[valid_positions].copy()
        y_train, y_valid = labels.iloc[train_positions], labels.iloc[valid_positions]

        governed_train, governed_valid, audit = learn_and_apply_governance(raw_train, raw_valid, ratio_columns)
        audit.insert(0, "fold", fold_number)
        audit_tables.append(audit)

        for experiment, x_train, x_valid in [
            ("raw_champion", raw_train, raw_valid),
            ("governed_champion", governed_train, governed_valid),
        ]:
            categorical = x_train.select_dtypes(include=["object", "category"]).columns.tolist()
            numeric = [column for column in x_train.columns if column not in categorical]
            preprocessor = build_preprocessor(numeric, categorical)
            transformed_train = preprocessor.fit_transform(x_train)
            transformed_valid = preprocessor.transform(x_valid)

            # 两组使用相同树数、参数和折内随机种子，保证差异只来自治理规则。
            model = build_cv_xgboost(config["random_state"] + fold_number, config["xgboost_cv_n_estimators"])
            model.fit(transformed_train, y_train)
            probabilities = model.predict_proba(transformed_valid)[:, 1]
            oof[experiment][valid_positions] = probabilities
            metrics = calculate_metrics(y_valid, probabilities)
            fold_rows.append({"experiment": experiment, "display_name": EXPERIMENT_NAMES[experiment], "fold": fold_number, **metrics})

    pd.concat(audit_tables, ignore_index=True).to_csv(output_dir / "fold_specific_governance_thresholds.csv", index=False)
    fold_results = pd.DataFrame(fold_rows)
    oof_table = pd.DataFrame({"actual_default": labels.to_numpy(), **{f"probability_{key}": value for key, value in oof.items()}})
    return fold_results, oof_table


def save_model_comparison(fold_results: pd.DataFrame, oof: pd.DataFrame, output_dir: Path) -> Dict:
    """汇总模型效果、五折胜出次数、校准指标和可视化。"""
    summary_rows = []
    for experiment in EXPERIMENT_NAMES:
        group = fold_results[fold_results["experiment"] == experiment]
        row = {"experiment": experiment, "display_name": EXPERIMENT_NAMES[experiment]}
        for metric in METRICS:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=1))
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)

    # 对每一折做配对差值；Brier越低越好，所以其改善方向要反转。
    pivot = fold_results.pivot(index="fold", columns="experiment")
    comparison_rows = []
    for metric in METRICS:
        raw_difference = pivot[metric]["governed_champion"] - pivot[metric]["raw_champion"]
        improvement = -raw_difference if metric == "brier_score" else raw_difference
        comparison_rows.append({
            "metric": metric, "mean_improvement": float(improvement.mean()),
            "improvement_std": float(improvement.std(ddof=1)),
            "governed_wins": int((improvement > 0).sum()), "total_folds": int(len(improvement)),
        })
    paired = pd.DataFrame(comparison_rows)
    summary.to_csv(output_dir / "model_metrics_summary.csv", index=False)
    fold_results.to_csv(output_dir / "model_metrics_by_fold.csv", index=False)
    paired.to_csv(output_dir / "paired_governance_improvement.csv", index=False)
    oof.to_csv(output_dir / "oof_predictions.csv", index=False)

    calibration_tables, calibration_rows = [], []
    labels = oof["actual_default"].to_numpy()
    for experiment in EXPERIMENT_NAMES:
        probabilities = oof[f"probability_{experiment}"].to_numpy()
        calibration_tables.append(make_calibration_table(labels, probabilities, experiment))
        calibration_intercept, calibration_slope = estimate_calibration_intercept_slope(labels, probabilities)
        calibration_rows.append({
            "experiment": experiment,
            "ece_10_equal_frequency_bins": expected_calibration_error(labels, probabilities),
            "calibration_intercept": calibration_intercept,
            "calibration_slope": calibration_slope,
            "mean_predicted_risk": float(probabilities.mean()),
            "actual_default_rate": float(labels.mean()),
        })
    calibration = pd.concat(calibration_tables, ignore_index=True)
    calibration_summary = pd.DataFrame(calibration_rows)
    calibration.to_csv(output_dir / "calibration_by_risk_decile.csv", index=False)
    calibration_summary.to_csv(output_dir / "calibration_summary.csv", index=False)

    figure, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    colors = {"raw_champion": "#2474B5", "governed_champion": "#D8792B"}
    for experiment in EXPERIMENT_NAMES:
        table = calibration[calibration["experiment"] == experiment]
        axes[0].plot(table["mean_predicted_risk"], table["actual_default_rate"], marker="o", linewidth=2, color=colors[experiment], label=PLOT_NAMES[experiment])
        axes[1].hist(oof[f"probability_{experiment}"], bins=50, alpha=0.48, density=True, color=colors[experiment], label=PLOT_NAMES[experiment])
    axes[0].plot([0, 0.7], [0, 0.7], "--", color="gray", label="Perfect calibration")
    axes[0].set(title="OOF probability calibration", xlabel="Mean predicted risk", ylabel="Actual default rate")
    axes[1].set(title="OOF predicted-risk distribution", xlabel="Predicted default probability", ylabel="Density")
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "oof_calibration_and_probability_distribution.png", dpi=180)
    plt.close(figure)

    decision = {
        "all_four_mean_metrics_improved": bool((paired["mean_improvement"] > 0).all()),
        "governed_wins_by_metric": dict(zip(paired["metric"], paired["governed_wins"])),
        "decision_rule": "仅当治理版四项五折均值全部改善时自动替换冠军；否则结合稳定性、校准和治理价值人工判断。",
    }
    with (output_dir / "governance_decision.json").open("w", encoding="utf-8") as file:
        json.dump(decision, file, ensure_ascii=False, indent=2)
    return decision


def main() -> None:
    """依次运行质量诊断、五折治理实验和OOF概率校准分析。"""
    config = load_config()
    output_dir = config["output_dir"] / "data_quality_governance"
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    logging.info("构建完整冠军特征表")
    train, test, labels, customer_ids = build_complete_tables(config)
    del customer_ids
    logging.info("诊断训练/测试缺失率漂移")
    save_missingness_diagnostics(train, test, output_dir)
    logging.info("诊断比率字段极端分布")
    _, ratio_columns = save_distribution_diagnostics(train, output_dir)

    logging.info("运行原始冠军与治理版冠军的严格配对五折实验")
    fold_results, oof = run_paired_cv(train, labels, ratio_columns, config, output_dir)
    decision = save_model_comparison(fold_results, oof, output_dir)
    logging.info("治理实验完成：%s", decision)


if __name__ == "__main__":
    main()
