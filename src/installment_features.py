"""将分期还款明细聚合为客户级内部履约特征，并正确处理拆分付款。"""

import logging
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


CUSTOMER_ID = "SK_ID_CURR"
PREVIOUS_ID = "SK_ID_PREV"


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """安全计算比例；分母为0时返回缺失，避免产生无穷值。"""
    return numerator / denominator.replace(0, np.nan)


def _flatten_columns(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """将聚合产生的两层列名整理为稳定、可读的单层列名。"""
    frame.columns = [
        CUSTOMER_ID if first == CUSTOMER_ID else f"{prefix}_{first}_{second}".upper()
        for first, second in frame.columns
    ]
    return frame


def load_installment_payments(history_data_dir: Path) -> pd.DataFrame:
    """读取还款明细，并使用合适数据类型控制1360万行数据的内存占用。"""
    path = history_data_dir / "installments_payments.csv"
    if not path.exists():
        raise FileNotFoundError(f"没有找到 installments_payments 文件：{path}")

    # 表中只有8列且全部需要；明确较紧凑的数据类型可减少后续分组时的内存压力。
    payments = pd.read_csv(
        path,
        dtype={
            PREVIOUS_ID: "int32",
            CUSTOMER_ID: "int32",
            "NUM_INSTALMENT_VERSION": "float32",
            "NUM_INSTALMENT_NUMBER": "int16",
            "DAYS_INSTALMENT": "float32",
            "DAYS_ENTRY_PAYMENT": "float32",
            "AMT_INSTALMENT": "float32",
            "AMT_PAYMENT": "float32",
        },
    )
    required = {
        PREVIOUS_ID, CUSTOMER_ID, "NUM_INSTALMENT_VERSION", "NUM_INSTALMENT_NUMBER",
        "DAYS_INSTALMENT", "DAYS_ENTRY_PAYMENT", "AMT_INSTALMENT", "AMT_PAYMENT",
    }
    missing = required.difference(payments.columns)
    if missing:
        raise ValueError(f"installments_payments.csv 缺少必要字段：{sorted(missing)}")
    logging.info(
        "Installments Payments 已读取：%s 条付款记录，%s 名客户",
        len(payments), payments[CUSTOMER_ID].nunique(),
    )
    return payments


def consolidate_split_payments(payments: pd.DataFrame) -> pd.DataFrame:
    """先合并同一期的拆分付款，得到真实的一期应还、实还和完成日期。"""
    installment_keys = [
        CUSTOMER_ID, PREVIOUS_ID, "NUM_INSTALMENT_VERSION", "NUM_INSTALMENT_NUMBER"
    ]

    # 同一期可能分多笔支付：计划日期和应还额通常重复，不能直接求和；
    # 实还额需要求和，最后付款日期用于判断该期何时真正完成付款。
    installment = (
        payments.groupby(installment_keys, observed=True, sort=False)
        .agg(
            DAYS_INSTALMENT=("DAYS_INSTALMENT", "max"),
            FIRST_PAYMENT_DAY=("DAYS_ENTRY_PAYMENT", "min"),
            LAST_PAYMENT_DAY=("DAYS_ENTRY_PAYMENT", "max"),
            AMT_INSTALMENT=("AMT_INSTALMENT", "max"),
            AMT_PAYMENT=("AMT_PAYMENT", "sum"),
            PAYMENT_ROW_COUNT=("AMT_PAYMENT", "size"),
        )
        .reset_index()
    )
    logging.info(
        "拆分付款合并完成：%s 条付款明细合并为 %s 个分期",
        len(payments), len(installment),
    )
    return installment


def aggregate_installment_features(payments: pd.DataFrame) -> pd.DataFrame:
    """从还款时间、还款金额、严重程度、近期表现和趋势构造客户级特征。"""
    installment = consolidate_split_payments(payments)
    del payments

    # 正值表示完成该期付款时已经逾期；提前还款产生负值，风险统计中截断为0。
    installment["PAYMENT_DELAY_DAYS_RAW"] = (
        installment["LAST_PAYMENT_DAY"] - installment["DAYS_INSTALMENT"]
    )
    installment["PAYMENT_DELAY_DAYS"] = installment[
        "PAYMENT_DELAY_DAYS_RAW"
    ].clip(lower=0)
    installment["EARLY_PAYMENT_DAYS"] = (
        installment["DAYS_INSTALMENT"] - installment["FIRST_PAYMENT_DAY"]
    ).clip(lower=0)

    # 少还金额为应还减实还；负值代表超额支付，因此分别截断生成少还与超额还款。
    installment["UNDERPAYMENT_AMOUNT"] = (
        installment["AMT_INSTALMENT"] - installment["AMT_PAYMENT"]
    ).clip(lower=0)
    installment["OVERPAYMENT_AMOUNT"] = (
        installment["AMT_PAYMENT"] - installment["AMT_INSTALMENT"]
    ).clip(lower=0)
    installment["PAYMENT_TO_DUE_RATIO"] = _safe_ratio(
        installment["AMT_PAYMENT"], installment["AMT_INSTALMENT"]
    )

    # 允许金额计算中极小的浮点误差；超过1个货币单位才标记为少还。
    installment["IS_LATE"] = (installment["PAYMENT_DELAY_DAYS"] > 0).astype("int8")
    installment["IS_UNDERPAID"] = (installment["UNDERPAYMENT_AMOUNT"] > 1).astype("int8")
    installment["IS_SPLIT_PAYMENT"] = (installment["PAYMENT_ROW_COUNT"] > 1).astype("int8")
    installment["IS_FULLY_PAID"] = (
        installment["AMT_PAYMENT"] + 1 >= installment["AMT_INSTALMENT"]
    ).astype("int8")

    # 不同逾期阈值区分偶尔晚1天与30/60/90天严重拖欠，避免被平均值掩盖。
    for days in (7, 30, 60, 90):
        installment[f"IS_LATE_OVER_{days}D"] = (
            installment["PAYMENT_DELAY_DAYS"] > days
        ).astype("int8")

    # DAYS_INSTALMENT相对当前申请日为负；越接近0的计划还款越近期。
    for months, days in ((3, 90), (6, 180), (12, 365), (24, 730)):
        installment[f"IS_RECENT_{months}M"] = (
            installment["DAYS_INSTALMENT"] >= -days
        ).astype("int8")
        installment[f"LATE_RECENT_{months}M"] = (
            (installment[f"IS_RECENT_{months}M"] == 1)
            & (installment["IS_LATE"] == 1)
        ).astype("int8")
        installment[f"UNDERPAID_RECENT_{months}M"] = (
            (installment[f"IS_RECENT_{months}M"] == 1)
            & (installment["IS_UNDERPAID"] == 1)
        ).astype("int8")

    aggregation: Dict[str, list] = {
        PREVIOUS_ID: ["nunique"],
        "NUM_INSTALMENT_NUMBER": ["count", "max", "mean"],
        "DAYS_INSTALMENT": ["min", "max", "mean", "std"],
        "PAYMENT_DELAY_DAYS": ["max", "mean", "sum", "std"],
        "PAYMENT_DELAY_DAYS_RAW": ["min", "max", "mean"],
        "EARLY_PAYMENT_DAYS": ["max", "mean"],
        "AMT_INSTALMENT": ["min", "max", "mean", "sum"],
        "AMT_PAYMENT": ["min", "max", "mean", "sum"],
        "UNDERPAYMENT_AMOUNT": ["max", "mean", "sum"],
        "OVERPAYMENT_AMOUNT": ["max", "mean", "sum"],
        "PAYMENT_TO_DUE_RATIO": ["min", "max", "mean", "std"],
        "PAYMENT_ROW_COUNT": ["max", "mean", "sum"],
        "IS_LATE": ["sum", "mean"],
        "IS_UNDERPAID": ["sum", "mean"],
        "IS_SPLIT_PAYMENT": ["sum", "mean"],
        "IS_FULLY_PAID": ["sum", "mean"],
        "IS_LATE_OVER_7D": ["sum", "mean"],
        "IS_LATE_OVER_30D": ["sum", "mean"],
        "IS_LATE_OVER_60D": ["sum", "mean"],
        "IS_LATE_OVER_90D": ["sum", "mean"],
    }
    for months in (3, 6, 12, 24):
        aggregation[f"IS_RECENT_{months}M"] = ["sum"]
        aggregation[f"LATE_RECENT_{months}M"] = ["sum"]
        aggregation[f"UNDERPAID_RECENT_{months}M"] = ["sum"]

    customer = installment.groupby(CUSTOMER_ID, observed=True).agg(aggregation).reset_index()
    customer = _flatten_columns(customer, "inst")

    # 最近窗口的事件数必须除以同期观察分期数，避免记录更多的人天然事件数更高。
    for months in (3, 6, 12, 24):
        observed = customer[f"INST_IS_RECENT_{months}M_SUM"]
        customer[f"INST_LATE_RATE_RECENT_{months}M"] = _safe_ratio(
            customer[f"INST_LATE_RECENT_{months}M_SUM"], observed
        )
        customer[f"INST_UNDERPAID_RATE_RECENT_{months}M"] = _safe_ratio(
            customer[f"INST_UNDERPAID_RECENT_{months}M_SUM"], observed
        )

    # 组合层面的实际支付/应还金额比率，与逐期比率均值表达不同的总体偿付完整度。
    customer["INST_TOTAL_PAYMENT_TO_DUE_RATIO"] = _safe_ratio(
        customer["INST_AMT_PAYMENT_SUM"], customer["INST_AMT_INSTALMENT_SUM"]
    )
    customer["HAS_INSTALLMENT_HISTORY"] = np.int8(1)
    return customer


def build_installment_features(history_data_dir: Path) -> pd.DataFrame:
    """读取还款明细并返回可直接连接主表的客户级履约特征。"""
    payments = load_installment_payments(history_data_dir)
    customer = aggregate_installment_features(payments)
    logging.info(
        "Installments 聚合完成：%s 名客户，%s 个还款行为特征",
        len(customer), customer.shape[1] - 1,
    )
    return customer
