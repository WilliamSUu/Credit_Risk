"""把 previous_application 历史申请表聚合成“一名客户一行”的可解释特征。"""

import logging
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


CUSTOMER_ID = "SK_ID_CURR"
PREVIOUS_ID = "SK_ID_PREV"


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """安全计算比率；分母为 0 时保留缺失，避免生成无穷大。"""
    return numerator / denominator.replace(0, np.nan)


def _flatten_columns(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """把 groupby 的两层列名整理成 PREV_字段_统计量 的单层名称。"""
    frame.columns = [
        CUSTOMER_ID if first == CUSTOMER_ID else f"{prefix}_{first}_{second}".upper()
        for first, second in frame.columns
    ]
    return frame


def load_previous_application(history_data_dir: Path) -> pd.DataFrame:
    """读取历史申请表，并检查客户键、申请键和状态字段是否完整。"""
    path = history_data_dir / "previous_application.csv"
    if not path.exists():
        raise FileNotFoundError(f"没有找到 previous_application 文件：{path}")

    previous = pd.read_csv(path)
    required = {CUSTOMER_ID, PREVIOUS_ID, "NAME_CONTRACT_STATUS", "DAYS_DECISION"}
    missing = required.difference(previous.columns)
    if missing:
        raise ValueError(f"previous_application.csv 缺少必要字段：{sorted(missing)}")

    # 字符串列的取值种类有限，转为 category 可明显降低约 167 万行数据的内存占用。
    for column in previous.select_dtypes(include="object").columns:
        previous[column] = previous[column].astype("category")
    logging.info(
        "Previous Application 已读取：%s 笔申请，%s 名客户",
        len(previous), previous[CUSTOMER_ID].nunique(),
    )
    return previous


def aggregate_previous_application(previous: pd.DataFrame) -> pd.DataFrame:
    """从频率、状态、金额、期限、渠道和时间六方面构造客户级历史申请特征。"""
    work = previous.copy()

    # 365243 是 Home Credit 用于表示“尚未发生/未知日期”的特殊编码，不是真实天数。
    date_columns = [
        "DAYS_FIRST_DRAWING", "DAYS_FIRST_DUE", "DAYS_LAST_DUE_1ST_VERSION",
        "DAYS_LAST_DUE", "DAYS_TERMINATION",
    ]
    for column in date_columns:
        work[column] = work[column].replace(365243, np.nan)

    # 贷款级比率刻画申请需求、最终授信和客户首付之间的关系。
    work["CREDIT_TO_APPLICATION_RATIO"] = _safe_ratio(
        work["AMT_CREDIT"], work["AMT_APPLICATION"]
    )
    work["DOWN_PAYMENT_TO_APPLICATION_RATIO"] = _safe_ratio(
        work["AMT_DOWN_PAYMENT"], work["AMT_APPLICATION"]
    )
    work["ANNUITY_TO_CREDIT_RATIO"] = _safe_ratio(
        work["AMT_ANNUITY"], work["AMT_CREDIT"]
    )
    work["CREDIT_APPLICATION_DIFFERENCE"] = (
        work["AMT_CREDIT"] - work["AMT_APPLICATION"]
    )

    # 将审批结果转成 0/1 后求和得到次数、求均值得到占全部历史申请的比例。
    status_map = {
        "APPROVED": "Approved",
        "REFUSED": "Refused",
        "CANCELED": "Canceled",
        "UNUSED": "Unused offer",
    }
    for feature_name, source_value in status_map.items():
        work[f"IS_{feature_name}"] = (
            work["NAME_CONTRACT_STATUS"] == source_value
        ).astype("int8")

    # DAYS_DECISION 越接近 0，申请发生得越近；窗口计数反映短期资金需求强度。
    for days in (90, 180, 365, 730):
        work[f"APPLICATION_WITHIN_{days}D"] = (
            work["DAYS_DECISION"] >= -days
        ).astype("int8")

    # 标记常见产品类型，既保留业务含义，也避免将高基数类别全部展开后再聚合。
    contract_types = {
        "CASH_LOAN": "Cash loans",
        "CONSUMER_LOAN": "Consumer loans",
        "REVOLVING_LOAN": "Revolving loans",
    }
    for feature_name, source_value in contract_types.items():
        work[f"IS_{feature_name}"] = (
            work["NAME_CONTRACT_TYPE"] == source_value
        ).astype("int8")

    # 全部历史申请的聚合用于描述总体申请行为和极端金额。
    aggregation: Dict[str, list] = {
        PREVIOUS_ID: ["count"],
        "DAYS_DECISION": ["min", "max", "mean", "std"],
        "AMT_ANNUITY": ["min", "max", "mean", "sum"],
        "AMT_APPLICATION": ["min", "max", "mean", "sum"],
        "AMT_CREDIT": ["min", "max", "mean", "sum"],
        "AMT_DOWN_PAYMENT": ["min", "max", "mean", "sum"],
        "AMT_GOODS_PRICE": ["min", "max", "mean", "sum"],
        "RATE_DOWN_PAYMENT": ["min", "max", "mean"],
        "CNT_PAYMENT": ["min", "max", "mean", "sum"],
        "HOUR_APPR_PROCESS_START": ["min", "max", "mean"],
        "SELLERPLACE_AREA": ["min", "max", "mean"],
        "CREDIT_TO_APPLICATION_RATIO": ["min", "max", "mean"],
        "DOWN_PAYMENT_TO_APPLICATION_RATIO": ["min", "max", "mean"],
        "ANNUITY_TO_CREDIT_RATIO": ["min", "max", "mean"],
        "CREDIT_APPLICATION_DIFFERENCE": ["min", "max", "mean", "sum"],
        "IS_APPROVED": ["sum", "mean"],
        "IS_REFUSED": ["sum", "mean"],
        "IS_CANCELED": ["sum", "mean"],
        "IS_UNUSED": ["sum", "mean"],
        "IS_CASH_LOAN": ["sum", "mean"],
        "IS_CONSUMER_LOAN": ["sum", "mean"],
        "IS_REVOLVING_LOAN": ["sum", "mean"],
        "APPLICATION_WITHIN_90D": ["sum"],
        "APPLICATION_WITHIN_180D": ["sum"],
        "APPLICATION_WITHIN_365D": ["sum"],
        "APPLICATION_WITHIN_730D": ["sum"],
    }
    customer = work.groupby(CUSTOMER_ID, observed=True).agg(aggregation).reset_index()
    customer = _flatten_columns(customer, "prev")

    # 产品、渠道和用途的种类数反映历史金融需求是否集中在单一场景。
    category_columns = [
        "NAME_CONTRACT_TYPE", "NAME_CASH_LOAN_PURPOSE", "NAME_PAYMENT_TYPE",
        "NAME_GOODS_CATEGORY", "NAME_PORTFOLIO", "NAME_PRODUCT_TYPE",
        "CHANNEL_TYPE", "NAME_SELLER_INDUSTRY", "NAME_YIELD_GROUP",
        "PRODUCT_COMBINATION", "CODE_REJECT_REASON",
    ]
    category_counts = (
        work.groupby(CUSTOMER_ID, observed=True)[category_columns]
        .nunique()
        .add_prefix("PREV_")
        .add_suffix("_NUNIQUE")
        .reset_index()
    )
    customer = customer.merge(
        category_counts, on=CUSTOMER_ID, how="left", validate="one_to_one"
    )

    # 只有已批准合同才存在实际执行日期；单独聚合可避免拒绝申请的日期缺失污染结果。
    approved = work.loc[work["IS_APPROVED"] == 1].copy()
    approved["PLANNED_DURATION_DAYS"] = (
        approved["DAYS_LAST_DUE_1ST_VERSION"] - approved["DAYS_FIRST_DUE"]
    )
    approved["ACTUAL_DURATION_DAYS"] = (
        approved["DAYS_TERMINATION"] - approved["DAYS_FIRST_DUE"]
    )
    approved["TERMINATION_VS_PLAN_DAYS"] = (
        approved["DAYS_TERMINATION"] - approved["DAYS_LAST_DUE_1ST_VERSION"]
    )
    approved_aggregation = {
        "DAYS_FIRST_DUE": ["min", "max", "mean"],
        "DAYS_LAST_DUE_1ST_VERSION": ["min", "max", "mean"],
        "DAYS_LAST_DUE": ["min", "max", "mean"],
        "DAYS_TERMINATION": ["min", "max", "mean"],
        "PLANNED_DURATION_DAYS": ["min", "max", "mean"],
        "ACTUAL_DURATION_DAYS": ["min", "max", "mean"],
        "TERMINATION_VS_PLAN_DAYS": ["min", "max", "mean"],
        "NFLAG_INSURED_ON_APPROVAL": ["sum", "mean"],
    }
    approved_customer = (
        approved.groupby(CUSTOMER_ID, observed=True)
        .agg(approved_aggregation)
        .reset_index()
    )
    approved_customer = _flatten_columns(approved_customer, "prev_approved")
    customer = customer.merge(
        approved_customer, on=CUSTOMER_ID, how="left", validate="one_to_one"
    )

    # 组合层面申请金额与授信金额的比例比逐笔平均更能反映历史总体批准程度。
    customer["PREV_TOTAL_CREDIT_TO_APPLICATION_RATIO"] = _safe_ratio(
        customer["PREV_AMT_CREDIT_SUM"], customer["PREV_AMT_APPLICATION_SUM"]
    )
    customer["HAS_PREVIOUS_APPLICATION_HISTORY"] = np.int8(1)
    return customer


def build_previous_application_features(history_data_dir: Path) -> pd.DataFrame:
    """读取并返回可直接连接到主申请表的客户级历史申请特征。"""
    previous = load_previous_application(history_data_dir)
    customer_features = aggregate_previous_application(previous)
    logging.info(
        "Previous Application 聚合完成：%s 名客户，%s 个特征",
        len(customer_features), customer_features.shape[1] - 1,
    )
    return customer_features
