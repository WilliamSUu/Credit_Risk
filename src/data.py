"""读取 Home Credit 主申请表，并在建模前检查最基本的数据结构。"""

from pathlib import Path
from typing import Tuple

import pandas as pd


TARGET_COLUMN = "TARGET"
ID_COLUMN = "SK_ID_CURR"


def load_application_data(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """读取训练集与测试集，并确认标签列、客户申请 ID 都存在。

    data_dir: 同时放有 application_train.csv 和 application_test.csv 的目录。
    返回值: (训练集 DataFrame, 测试集 DataFrame)。历史行为表由各自的特征模块读取，
    聚合到客户层级后再通过 SK_ID_CURR 与这里的主表连接。
    """
    # 用配置的数据目录拼出两个主表的完整路径。
    train_path = data_dir / "application_train.csv"
    test_path = data_dir / "application_test.csv"

    # 先检查文件是否存在；这样路径写错时能立刻得到可读错误，而不是模糊的读文件报错。
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            "Expected application_train.csv and application_test.csv under "
            "the configured data_dir: {}".format(data_dir)
        )

    # 这里仅负责主表；一行对应一位申请人/一笔当前申请。
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    # 训练集必须带有 TARGET；测试集没有 TARGET，这是比赛数据的正常设计。
    if TARGET_COLUMN not in train.columns:
        raise ValueError("Training data is missing the TARGET column.")
    # 两张表都应有 SK_ID_CURR，用于识别申请记录，但该 ID 不作为预测变量。
    if ID_COLUMN not in train.columns or ID_COLUMN not in test.columns:
        raise ValueError("Train and test data must both contain SK_ID_CURR.")

    return train, test
