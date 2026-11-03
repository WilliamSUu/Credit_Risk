"""从主申请表准备模型输入，并建立训练集内拟合的预处理流程。"""

from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ID_COLUMN = "SK_ID_CURR"
TARGET_COLUMN = "TARGET"


def make_features(
    train: pd.DataFrame, test: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, List[str], List[str]]:
    """拆出标签、删除 ID，并识别数值/类别字段。

    注意：当前模型只使用训练集做验证，因此返回测试特征仅用于确保字段结构一致。
    返回顺序：训练特征、测试特征、训练标签、数值列名、类别列名。
    """
    # TARGET 是我们要预测的答案，不能放进输入，否则模型会直接“偷看答案”。
    labels = train[TARGET_COLUMN].astype("int8")
    # SK_ID_CURR 只是记录编号，没有可泛化的风险含义，所以从特征中移除。
    train_features = train.drop(columns=[TARGET_COLUMN, ID_COLUMN], errors="ignore")
    test_features = test.drop(columns=[ID_COLUMN], errors="ignore")
    # 让训练表和测试表列名一致；测试集缺少的列会补为空，额外列也会被显式纳入结构。
    train_features, test_features = train_features.align(
        test_features, join="outer", axis=1, copy=False
    )

    # 字符串/类别字段需要独热编码；其它字段按数值处理。
    categorical_columns = train_features.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()
    numeric_columns = [
        column for column in train_features.columns
        if column not in categorical_columns
    ]

    # DAYS_EMPLOYED 中的 365243 是数据集用来表示“退休/特殊缺失”的哨兵值，
    # 不是实际工作了 365243 天；将它视为缺失，避免模型把它误解成极长工龄。
    for frame in (train_features, test_features):
        if "DAYS_EMPLOYED" in frame.columns:
            frame["DAYS_EMPLOYED"] = frame["DAYS_EMPLOYED"].replace(365243, np.nan)

    return (
        train_features,
        test_features,
        labels,
        numeric_columns,
        categorical_columns,
    )


def build_preprocessor(
    numeric_columns: List[str], categorical_columns: List[str]
) -> ColumnTransformer:
    """建立数值列和类别列各自的预处理流水线。

    ColumnTransformer 会按列类型分别处理，再把结果拼成模型可以读取的矩阵。
    预处理器放入总 Pipeline 后，填补值和缩放参数只从训练折学习，减少验证泄漏。
    """
    # 数值列先用训练样本中位数填缺失，再为存在缺失的列增加缺失标记，最后标准化。
    numeric_pipeline = Pipeline(
        steps=[
            # 中位数不易被极端值带偏；add_indicator 保留“原来缺失”这一信息。
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            # 标准化让不同量纲的数值更适合逻辑回归的正则化优化。
            ("scaler", StandardScaler()),
        ]
    )
    # 类别列先填最常见类别，再转成 0/1 指示列（独热编码）。
    categorical_pipeline = Pipeline(
        steps=[
            # 类别缺失用训练数据中最常见的类别暂时补齐。
            ("imputer", SimpleImputer(strategy="most_frequent")),
            # handle_unknown="ignore" 可容忍验证/未来数据出现训练中没见过的类别。
            ("one_hot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    # 按列名选择对应处理器；其余列丢弃，避免不小心把非特征列送进模型。
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
        sparse_threshold=1.0,
    )
