"""将 Bureau 外部征信贷款和月度状态聚合为“一名客户一行”的模型特征。"""

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


CUSTOMER_ID = "SK_ID_CURR"
BUREAU_ID = "SK_ID_BUREAU"


def _flatten_columns(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """把 groupby 产生的两层列名改成稳定、可读的普通列名。"""
    frame.columns = [
        CUSTOMER_ID if first == CUSTOMER_ID else f"{prefix}_{first}_{second}".upper()
        for first, second in frame.columns
    ]
    return frame


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """安全计算比例；分母为 0 时返回缺失，避免产生正负无穷大。"""
    denominator = denominator.replace(0, np.nan)
    return numerator / denominator


def load_bureau(history_data_dir: Path) -> pd.DataFrame:
    """读取贷款级 Bureau 表，并执行必要的异常值与类型处理。"""
    path = history_data_dir / "bureau.csv"
    if not path.exists():
        raise FileNotFoundError(f"没有找到 Bureau 文件：{path}")

    # Bureau 约 171 万行；一次读取能够保留贷款之间的完整客户分组关系。
    bureau = pd.read_csv(path)
    required = {CUSTOMER_ID, BUREAU_ID, "CREDIT_ACTIVE", "DAYS_CREDIT"}
    missing = required.difference(bureau.columns)
    if missing:
        raise ValueError(f"bureau.csv 缺少必要字段：{sorted(missing)}")

    # 将有限取值的字符串列转为 category，明显减少聚合期间的内存占用。
    for column in ("CREDIT_ACTIVE", "CREDIT_CURRENCY", "CREDIT_TYPE"):
        bureau[column] = bureau[column].astype("category")
    return bureau


def aggregate_bureau(bureau: pd.DataFrame) -> pd.DataFrame:
    """从贷款数量、状态、金额、逾期和时间五方面生成客户级外部征信特征。"""
    work = bureau.copy()

    # 贷款级比例比单独金额更能体现偿债压力；异常分母保留为缺失交给预处理器。
    work["DEBT_TO_CREDIT_RATIO"] = _safe_ratio(
        work["AMT_CREDIT_SUM_DEBT"], work["AMT_CREDIT_SUM"]
    )
    work["OVERDUE_TO_DEBT_RATIO"] = _safe_ratio(
        work["AMT_CREDIT_SUM_OVERDUE"], work["AMT_CREDIT_SUM_DEBT"]
    )

    # 状态和关键事件先变成 0/1，再按客户求和与均值；均值自然等于发生比例。
    work["IS_ACTIVE"] = (work["CREDIT_ACTIVE"] == "Active").astype("int8")
    work["IS_CLOSED"] = (work["CREDIT_ACTIVE"] == "Closed").astype("int8")
    work["IS_SOLD"] = (work["CREDIT_ACTIVE"] == "Sold").astype("int8")
    work["IS_BAD_DEBT"] = (work["CREDIT_ACTIVE"] == "Bad debt").astype("int8")
    work["HAS_OVERDUE"] = (work["CREDIT_DAY_OVERDUE"].fillna(0) > 0).astype("int8")
    work["HAS_OVERDUE_AMOUNT"] = (work["AMT_CREDIT_SUM_OVERDUE"].fillna(0) > 0).astype("int8")

    # DAYS_CREDIT 越接近 0 表示借款越新；窗口计数反映近期集中申请外部信贷。
    for days in (90, 180, 365, 730):
        work[f"CREDIT_WITHIN_{days}D"] = (work["DAYS_CREDIT"] >= -days).astype("int8")

    # 对每一类数值选择有业务意义的统计量，而不是机械生成所有组合。
    aggregation = {
        BUREAU_ID: ["count"],
        "DAYS_CREDIT": ["min", "max", "mean", "std"],
        "CREDIT_DAY_OVERDUE": ["max", "mean", "sum"],
        "DAYS_CREDIT_ENDDATE": ["min", "max", "mean"],
        "DAYS_ENDDATE_FACT": ["min", "max", "mean"],
        "AMT_CREDIT_MAX_OVERDUE": ["max", "mean", "sum"],
        "CNT_CREDIT_PROLONG": ["max", "mean", "sum"],
        "AMT_CREDIT_SUM": ["min", "max", "mean", "sum"],
        "AMT_CREDIT_SUM_DEBT": ["max", "mean", "sum"],
        "AMT_CREDIT_SUM_LIMIT": ["max", "mean", "sum"],
        "AMT_CREDIT_SUM_OVERDUE": ["max", "mean", "sum"],
        "AMT_ANNUITY": ["max", "mean", "sum"],
        "DAYS_CREDIT_UPDATE": ["min", "max", "mean"],
        "DEBT_TO_CREDIT_RATIO": ["max", "mean"],
        "OVERDUE_TO_DEBT_RATIO": ["max", "mean"],
        "IS_ACTIVE": ["sum", "mean"],
        "IS_CLOSED": ["sum", "mean"],
        "IS_SOLD": ["sum", "mean"],
        "IS_BAD_DEBT": ["sum", "mean"],
        "HAS_OVERDUE": ["sum", "mean"],
        "HAS_OVERDUE_AMOUNT": ["sum", "mean"],
        "CREDIT_WITHIN_90D": ["sum"],
        "CREDIT_WITHIN_180D": ["sum"],
        "CREDIT_WITHIN_365D": ["sum"],
        "CREDIT_WITHIN_730D": ["sum"],
    }
    customer = work.groupby(CUSTOMER_ID, observed=True).agg(aggregation).reset_index()
    customer = _flatten_columns(customer, "bureau")

    # 不同 CREDIT_TYPE 的数量比完整独热编码更紧凑，同时保留产品多样性信息。
    credit_type_count = (
        work.groupby(CUSTOMER_ID, observed=True)["CREDIT_TYPE"]
        .nunique()
        .rename("BUREAU_CREDIT_TYPE_NUNIQUE")
        .reset_index()
    )
    customer = customer.merge(credit_type_count, on=CUSTOMER_ID, how="left", validate="one_to_one")

    # “当前仍活跃”的贷款更接近申请时负债压力，所以额外生成活跃贷款金额摘要。
    active = work.loc[work["IS_ACTIVE"] == 1]
    active_aggregation = {
        "AMT_CREDIT_SUM": ["sum", "mean", "max"],
        "AMT_CREDIT_SUM_DEBT": ["sum", "mean", "max"],
        "AMT_CREDIT_SUM_OVERDUE": ["sum", "max"],
        "AMT_ANNUITY": ["sum", "mean"],
        "DAYS_CREDIT_ENDDATE": ["min", "max", "mean"],
    }
    active_customer = active.groupby(CUSTOMER_ID, observed=True).agg(active_aggregation).reset_index()
    active_customer = _flatten_columns(active_customer, "bureau_active")
    customer = customer.merge(active_customer, on=CUSTOMER_ID, how="left", validate="one_to_one")

    # 用“客户总债务/总授信”形成组合层面的利用率，区别于逐笔比例的平均值。
    customer["BUREAU_TOTAL_DEBT_TO_CREDIT_RATIO"] = _safe_ratio(
        customer["BUREAU_AMT_CREDIT_SUM_DEBT_SUM"],
        customer["BUREAU_AMT_CREDIT_SUM_SUM"],
    )
    customer["HAS_BUREAU_HISTORY"] = np.int8(1)
    return customer


def aggregate_bureau_balance(
    history_data_dir: Path, bureau_id_map: pd.DataFrame
) -> pd.DataFrame:
    """先把 2730 万条月度状态聚合到贷款，再聚合到客户，防止主表行数膨胀。"""
    path = history_data_dir / "bureau_balance.csv"
    if not path.exists():
        raise FileNotFoundError(f"没有找到 Bureau Balance 文件：{path}")

    # 只读取三个必要字段；STATUS 使用 category，避免 2730 万个 Python 字符串占用巨量内存。
    balance = pd.read_csv(
        path,
        usecols=[BUREAU_ID, "MONTHS_BALANCE", "STATUS"],
        dtype={BUREAU_ID: "int32", "MONTHS_BALANCE": "int16", "STATUS": "category"},
    )
    logging.info("Bureau Balance 已读取：%s 条月度记录", len(balance))

    # 0 表示当月正常，1-5 表示逐级加重的逾期；C 为关闭，X 为未知状态。
    delinquency_map = {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5}
    balance["DPD_LEVEL"] = balance["STATUS"].map(delinquency_map).astype("float32")
    balance["IS_DPD"] = balance["STATUS"].isin(["1", "2", "3", "4", "5"]).astype("int8")
    balance["IS_SEVERE_DPD"] = balance["STATUS"].isin(["3", "4", "5"]).astype("int8")
    balance["IS_CLOSED_MONTH"] = (balance["STATUS"] == "C").astype("int8")
    balance["IS_UNKNOWN_MONTH"] = (balance["STATUS"] == "X").astype("int8")

    # 最近窗口以 MONTHS_BALANCE 接近 0 为准；例如 -5 到 0 属于最近 6 个月。
    for months in (6, 12, 24):
        recent_mask = balance["MONTHS_BALANCE"] >= -(months - 1)
        balance[f"DPD_RECENT_{months}M"] = (recent_mask & (balance["IS_DPD"] == 1)).astype("int8")
        balance[f"OBS_RECENT_{months}M"] = recent_mask.astype("int8")

    loan_aggregation = {
        "MONTHS_BALANCE": ["count", "min", "max"],
        "DPD_LEVEL": ["max", "mean"],
        "IS_DPD": ["sum", "mean"],
        "IS_SEVERE_DPD": ["sum", "mean"],
        "IS_CLOSED_MONTH": ["sum", "mean"],
        "IS_UNKNOWN_MONTH": ["sum", "mean"],
        "DPD_RECENT_6M": ["sum"],
        "OBS_RECENT_6M": ["sum"],
        "DPD_RECENT_12M": ["sum"],
        "OBS_RECENT_12M": ["sum"],
        "DPD_RECENT_24M": ["sum"],
        "OBS_RECENT_24M": ["sum"],
    }
    loan = balance.groupby(BUREAU_ID, observed=True).agg(loan_aggregation)
    loan.columns = [f"BB_{first}_{second}".upper() for first, second in loan.columns]
    loan = loan.reset_index()
    del balance

    # 计算各贷款最近窗口逾期率；同时保留观察月数，区分历史长短不同的客户。
    for months in (6, 12, 24):
        loan[f"BB_DPD_RATE_RECENT_{months}M"] = _safe_ratio(
            loan[f"BB_DPD_RECENT_{months}M_SUM"],
            loan[f"BB_OBS_RECENT_{months}M_SUM"],
        )

    # 一条 Bureau 贷款只属于一个客户；validate 可在键关系异常时立即报错。
    loan = loan.merge(
        bureau_id_map.drop_duplicates(BUREAU_ID),
        on=BUREAU_ID,
        how="inner",
        validate="one_to_one",
    )

    # 贷款级月度摘要再次按客户聚合：sum 表示总风险事件，mean/max 表示典型和最坏合同。
    value_columns = [column for column in loan.columns if column not in (BUREAU_ID, CUSTOMER_ID)]
    aggregation = {}
    for column in value_columns:
        if column.endswith("_SUM") or column == "BB_MONTHS_BALANCE_COUNT":
            aggregation[column] = ["sum", "mean", "max"]
        else:
            aggregation[column] = ["mean", "max"]

    customer = loan.groupby(CUSTOMER_ID).agg(aggregation).reset_index()
    customer = _flatten_columns(customer, "bureau_balance")
    customer["BUREAU_BALANCE_LOAN_COUNT"] = loan.groupby(CUSTOMER_ID)[BUREAU_ID].nunique().values
    customer["HAS_BUREAU_BALANCE_HISTORY"] = np.int8(1)
    return customer


def build_external_credit_feature_sets(
    history_data_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """一次构建 Bureau 版和 Bureau+Balance 版，供相同交叉验证折公平比较。"""
    bureau = load_bureau(history_data_dir)
    logging.info("Bureau 已读取：%s 笔贷款，%s 名客户", len(bureau), bureau[CUSTOMER_ID].nunique())
    bureau_features = aggregate_bureau(bureau)

    id_map = bureau[[BUREAU_ID, CUSTOMER_ID]].copy()
    balance_features = aggregate_bureau_balance(history_data_dir, id_map)
    combined_features = bureau_features.merge(
        balance_features, on=CUSTOMER_ID, how="left", validate="one_to_one"
    )
    logging.info(
        "外部征信聚合完成：Bureau %s 个特征；加入 Balance 后 %s 个特征",
        bureau_features.shape[1] - 1,
        combined_features.shape[1] - 1,
    )
    return bureau_features, combined_features
