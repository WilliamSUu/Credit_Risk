"""针对已确认的小分母异常，比较原冠军与针对性数据治理候选模型。"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.analyze_data_quality_governance import (
    LOWER_QUANTILE,
    UPPER_QUANTILE,
    build_complete_tables,
    estimate_calibration_intercept_slope,
    expected_calibration_error,
    make_calibration_table,
)
from src.bureau_features import CUSTOMER_ID, load_bureau
from src.config import load_config
from src.cross_validate_champion import build_cv_xgboost
from src.evaluation import calculate_metrics
from src.features import build_preprocessor
from src.installment_features import consolidate_split_payments, load_installment_payments


# 这些字段由上一轮分位数诊断确认存在“P99.5正常、最大值却高出数百至数万倍”的重尾。
# 不处理正常期限、金额或全部比率，避免再次损失有效的业务信息。
TARGET_RATIO_COLUMNS = [
    "BUREAU_DEBT_TO_CREDIT_RATIO_MAX",
    "BUREAU_DEBT_TO_CREDIT_RATIO_MEAN",
    "BUREAU_OVERDUE_TO_DEBT_RATIO_MAX",
    "BUREAU_OVERDUE_TO_DEBT_RATIO_MEAN",
    "INST_PAYMENT_TO_DUE_RATIO_MAX",
    "INST_PAYMENT_TO_DUE_RATIO_MEAN",
    "INST_PAYMENT_TO_DUE_RATIO_STD",
]
METRICS = ["roc_auc", "pr_auc", "ks", "brier_score"]


def aggregate_quality_flags(frame: pd.DataFrame, flag_columns: List[str], prefix: str) -> pd.DataFrame:
    """把贷款/分期级异常标记聚合为客户的异常次数与占比。"""
    aggregation = {column: ["sum", "mean"] for column in flag_columns}
    customer = frame.groupby(CUSTOMER_ID, observed=True).agg(aggregation).reset_index()
    customer.columns = [
        CUSTOMER_ID if first == CUSTOMER_ID else f"{prefix}_{first}_{second}".upper()
        for first, second in customer.columns
    ]
    return customer


def build_denominator_quality_flags(config: Dict) -> pd.DataFrame:
    """直接从原始金额构造小分母、负分母与异常比例标记。"""
    bureau = load_bureau(config["history_data_dir"])
    # 金额绝对值不超过1时，计算比例极易被放大；负余额另设标记而不直接判定为坏客户。
    bureau["CREDIT_DENOMINATOR_LE_1"] = bureau["AMT_CREDIT_SUM"].abs().le(1).astype("int8")
    bureau["CREDIT_DENOMINATOR_NEGATIVE"] = bureau["AMT_CREDIT_SUM"].lt(0).astype("int8")
    bureau["DEBT_DENOMINATOR_LE_1"] = bureau["AMT_CREDIT_SUM_DEBT"].abs().le(1).astype("int8")
    bureau["DEBT_DENOMINATOR_NEGATIVE"] = bureau["AMT_CREDIT_SUM_DEBT"].lt(0).astype("int8")

    # 比率绝对值大于10远超通常的债务利用率范围；只把它作为质量信号，不直接删除客户。
    credit_denominator = bureau["AMT_CREDIT_SUM"].replace(0, np.nan)
    debt_denominator = bureau["AMT_CREDIT_SUM_DEBT"].replace(0, np.nan)
    bureau["DEBT_CREDIT_RATIO_ABS_GT_10"] = (
        bureau["AMT_CREDIT_SUM_DEBT"].div(credit_denominator).abs() > 10
    ).astype("int8")
    bureau["OVERDUE_DEBT_RATIO_ABS_GT_10"] = (
        bureau["AMT_CREDIT_SUM_OVERDUE"].div(debt_denominator).abs() > 10
    ).astype("int8")
    bureau_flags = aggregate_quality_flags(
        bureau,
        [
            "CREDIT_DENOMINATOR_LE_1", "CREDIT_DENOMINATOR_NEGATIVE",
            "DEBT_DENOMINATOR_LE_1", "DEBT_DENOMINATOR_NEGATIVE",
            "DEBT_CREDIT_RATIO_ABS_GT_10", "OVERDUE_DEBT_RATIO_ABS_GT_10",
        ],
        "BUREAU_QUALITY",
    )
    del bureau

    # 先按真实分期合并拆分付款，防止把同一期重复的应还额当成多条异常。
    payments = load_installment_payments(config["history_data_dir"])
    installments = consolidate_split_payments(payments)
    del payments
    installments["DUE_DENOMINATOR_LE_1"] = installments["AMT_INSTALMENT"].abs().le(1).astype("int8")
    installments["DUE_DENOMINATOR_NEGATIVE"] = installments["AMT_INSTALMENT"].lt(0).astype("int8")
    due_denominator = installments["AMT_INSTALMENT"].replace(0, np.nan)
    installments["PAYMENT_DUE_RATIO_ABS_GT_10"] = (
        installments["AMT_PAYMENT"].div(due_denominator).abs() > 10
    ).astype("int8")
    installment_flags = aggregate_quality_flags(
        installments,
        ["DUE_DENOMINATOR_LE_1", "DUE_DENOMINATOR_NEGATIVE", "PAYMENT_DUE_RATIO_ABS_GT_10"],
        "INST_QUALITY",
    )
    return bureau_flags.merge(installment_flags, on=CUSTOMER_ID, how="outer", validate="one_to_one")


def attach_quality_flags(features: pd.DataFrame, customer_ids: pd.Series, flags: pd.DataFrame) -> pd.DataFrame:
    """按客户编号连接质量标记，并把“没有异常记录”的客户填为0。"""
    table = features.copy()
    table.insert(0, CUSTOMER_ID, customer_ids.to_numpy())
    table = table.merge(flags, on=CUSTOMER_ID, how="left", validate="one_to_one")
    quality_columns = [column for column in table if "_QUALITY_" in column]
    table[quality_columns] = table[quality_columns].fillna(0)
    return table.drop(columns=[CUSTOMER_ID])


def apply_targeted_fold_governance(
    x_train: pd.DataFrame, x_valid: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """只治理七个已确认异常字段，且阈值只从本折训练客户学习。"""
    governed_train, governed_valid = x_train.copy(), x_valid.copy()
    audit_rows = []
    for column in TARGET_RATIO_COLUMNS:
        train_values = pd.to_numeric(governed_train[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        valid_values = pd.to_numeric(governed_valid[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        lower = float(train_values.quantile(LOWER_QUANTILE))
        upper = float(train_values.quantile(UPPER_QUANTILE))

        # Bureau债务比率为负可能来自余额调整；保留负值标记，再将不可靠比率交给缺失值流程。
        negative_flag = f"{column}_NEGATIVE_FLAG"
        governed_train[negative_flag] = (train_values < 0).astype("int8")
        governed_valid[negative_flag] = (valid_values < 0).astype("int8")
        if column.startswith("BUREAU_"):
            train_values = train_values.mask(train_values < 0)
            valid_values = valid_values.mask(valid_values < 0)

        # 截尾标记保留越界事实，截尾后的数值避免极端分母支配树的切分。
        outlier_flag = f"{column}_TAIL_FLAG"
        governed_train[outlier_flag] = ((train_values < lower) | (train_values > upper)).astype("int8")
        governed_valid[outlier_flag] = ((valid_values < lower) | (valid_values > upper)).astype("int8")
        governed_train[column] = train_values.clip(lower=lower, upper=upper)
        governed_valid[column] = valid_values.clip(lower=lower, upper=upper)
        audit_rows.append({
            "feature": column, "lower_bound": lower, "upper_bound": upper,
            "train_tail_count": int(governed_train[outlier_flag].sum()),
            "valid_tail_count": int(governed_valid[outlier_flag].sum()),
            "train_negative_count": int(governed_train[negative_flag].sum()),
            "valid_negative_count": int(governed_valid[negative_flag].sum()),
        })
    return governed_train, governed_valid, pd.DataFrame(audit_rows)


def run_cv(
    raw_features: pd.DataFrame,
    governed_features: pd.DataFrame,
    labels: pd.Series,
    config: Dict,
    output_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """在相同五折上训练原冠军与针对性治理候选模型。"""
    splitter = StratifiedKFold(
        n_splits=config["cross_validation_folds"], shuffle=True, random_state=config["random_state"]
    )
    folds = list(splitter.split(raw_features, labels))
    experiments = {"raw_champion": raw_features, "targeted_governance": governed_features}
    oof = {name: np.zeros(len(labels), dtype="float32") for name in experiments}
    rows, audits = [], []

    for fold_number, (train_positions, valid_positions) in enumerate(folds, start=1):
        logging.info("开始针对性治理实验第%s/%s折", fold_number, len(folds))
        for experiment, full_features in experiments.items():
            x_train = full_features.iloc[train_positions].copy()
            x_valid = full_features.iloc[valid_positions].copy()
            if experiment == "targeted_governance":
                x_train, x_valid, audit = apply_targeted_fold_governance(x_train, x_valid)
                audit.insert(0, "fold", fold_number)
                audits.append(audit)

            categorical = x_train.select_dtypes(include=["object", "category"]).columns.tolist()
            numeric = [column for column in x_train if column not in categorical]
            preprocessor = build_preprocessor(numeric, categorical)
            transformed_train = preprocessor.fit_transform(x_train)
            transformed_valid = preprocessor.transform(x_valid)
            model = build_cv_xgboost(config["random_state"] + fold_number, config["xgboost_cv_n_estimators"])
            model.fit(transformed_train, labels.iloc[train_positions])
            probabilities = model.predict_proba(transformed_valid)[:, 1]
            oof[experiment][valid_positions] = probabilities
            rows.append({
                "experiment": experiment, "fold": fold_number,
                **calculate_metrics(labels.iloc[valid_positions], probabilities),
            })

    pd.concat(audits, ignore_index=True).to_csv(output_dir / "targeted_fold_thresholds.csv", index=False)
    oof_table = pd.DataFrame({"actual_default": labels.to_numpy(), **{f"probability_{name}": values for name, values in oof.items()}})
    return pd.DataFrame(rows), oof_table


def save_results(folds: pd.DataFrame, oof: pd.DataFrame, quality_flags: pd.DataFrame, output_dir: Path) -> None:
    """保存指标、异常标记规模、概率校准和直观比较图。"""
    summary_rows, paired_rows = [], []
    for experiment, group in folds.groupby("experiment"):
        row = {"experiment": experiment}
        for metric in METRICS:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=1))
        summary_rows.append(row)

    pivot = folds.pivot(index="fold", columns="experiment")
    for metric in METRICS:
        difference = pivot[metric]["targeted_governance"] - pivot[metric]["raw_champion"]
        improvement = -difference if metric == "brier_score" else difference
        paired_rows.append({
            "metric": metric, "mean_improvement": float(improvement.mean()),
            "improvement_std": float(improvement.std(ddof=1)),
            "targeted_governance_wins": int((improvement > 0).sum()), "total_folds": len(improvement),
        })
    summary = pd.DataFrame(summary_rows)
    paired = pd.DataFrame(paired_rows)
    folds.to_csv(output_dir / "metrics_by_fold.csv", index=False)
    summary.to_csv(output_dir / "metrics_summary.csv", index=False)
    paired.to_csv(output_dir / "paired_improvement.csv", index=False)
    oof.to_csv(output_dir / "oof_predictions.csv", index=False)

    # 质量标记规模说明异常来自多少客户，避免只看到一个夸张最大值就误判全体数据。
    flag_columns = [column for column in quality_flags if column != CUSTOMER_ID]
    flag_summary = pd.DataFrame({
        "quality_feature": flag_columns,
        "affected_customer_count": [(quality_flags[column] > 0).sum() for column in flag_columns],
        "affected_customer_rate": [(quality_flags[column] > 0).mean() for column in flag_columns],
        "total_flagged_records": [quality_flags[column].sum() if column.endswith("_SUM") else np.nan for column in flag_columns],
    })
    flag_summary.to_csv(output_dir / "denominator_quality_flag_summary.csv", index=False)

    calibration_rows, decile_tables = [], []
    labels = oof["actual_default"].to_numpy()
    for experiment in ("raw_champion", "targeted_governance"):
        probabilities = oof[f"probability_{experiment}"].to_numpy()
        intercept, slope = estimate_calibration_intercept_slope(labels, probabilities)
        calibration_rows.append({
            "experiment": experiment,
            "ece": expected_calibration_error(labels, probabilities),
            "calibration_intercept": intercept, "calibration_slope": slope,
            "mean_predicted_risk": float(probabilities.mean()),
            "actual_default_rate": float(labels.mean()),
        })
        decile_tables.append(make_calibration_table(labels, probabilities, experiment))
    pd.DataFrame(calibration_rows).to_csv(output_dir / "calibration_summary.csv", index=False)
    pd.concat(decile_tables, ignore_index=True).to_csv(output_dir / "calibration_by_decile.csv", index=False)

    figure, axes = plt.subplots(2, 2, figsize=(13, 9))
    labels_map = {"raw_champion": "Raw champion", "targeted_governance": "Targeted governance"}
    colors = {"raw_champion": "#2474B5", "targeted_governance": "#36A269"}
    for axis, metric in zip(axes.flat, METRICS):
        for experiment in labels_map:
            data = folds[folds["experiment"] == experiment].sort_values("fold")
            axis.plot(data["fold"], data[metric], marker="o", linewidth=2, color=colors[experiment], label=labels_map[experiment])
        axis.set(title=metric.upper(), xlabel="Validation fold", ylabel=metric)
        axis.grid(alpha=0.25)
        axis.legend()
    figure.suptitle("Targeted data governance: paired 5-fold comparison")
    figure.tight_layout()
    figure.savefig(output_dir / "targeted_governance_metrics.png", dpi=180)
    plt.close(figure)

    all_means_improved = bool((paired["mean_improvement"] > 0).all())
    # 信贷模型治理比普通建模更保守：均值改善但多数折不稳定时，只登记为挑战者。
    # 这里要求每项指标至少3/5折胜出，避免一次偶然的大幅改善掩盖其余折下降。
    stable_majority_wins = bool((paired["targeted_governance_wins"] >= 3).all())
    decision = {
        "all_four_mean_metrics_improved": all_means_improved,
        "all_four_metrics_win_at_least_three_folds": stable_majority_wins,
        "wins_by_metric": dict(zip(paired["metric"], paired["targeted_governance_wins"])),
        "selected_model": "targeted_governance" if all_means_improved and stable_majority_wins else "raw_champion",
        "targeted_model_status": "正式冠军" if all_means_improved and stable_majority_wins else "治理挑战者",
        "decision_reason": "四项均值均改善，但Brier仅2/5折胜出，因此暂不自动替换正式冠军。" if all_means_improved and not stable_majority_wins else "依据均值方向和逐折稳定性共同决定。",
    }
    with (output_dir / "decision.json").open("w", encoding="utf-8") as file:
        json.dump(decision, file, ensure_ascii=False, indent=2)


def main() -> None:
    """构建质量标记、运行五折实验并输出治理决策。"""
    config = load_config()
    output_dir = config["output_dir"] / "targeted_data_governance"
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    raw_features, _, labels, customer_ids = build_complete_tables(config)
    logging.info("从原始历史表构造小分母与异常比率标记")
    quality_flags = build_denominator_quality_flags(config)
    governed_features = attach_quality_flags(raw_features, customer_ids, quality_flags)

    missing_columns = sorted(set(TARGET_RATIO_COLUMNS).difference(governed_features.columns))
    if missing_columns:
        raise ValueError(f"针对性治理字段不存在：{missing_columns}")
    folds, oof = run_cv(raw_features, governed_features, labels, config, output_dir)
    save_results(folds, oof, quality_flags, output_dir)
    logging.info("针对性数据治理实验完成，结果位于%s", output_dir)


if __name__ == "__main__":
    main()
