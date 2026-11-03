"""读取项目配置，并把配置中的路径转换为可直接使用的 Path 对象。"""

from pathlib import Path
from typing import Any, Dict

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
# PROJECT_ROOT 指向仓库根目录，不依赖 PyCharm 当前把哪个目录设为工作目录。


def load_config(config_path: Path = PROJECT_ROOT / "config.yaml") -> Dict[str, Any]:
    """加载 YAML 配置，并返回包含绝对数据路径和输出路径的字典。

    config_path: 配置文件的位置；默认读取仓库根目录下的 config.yaml。
    返回值: 普通 Python 字典，路径项目会被转换为 pathlib.Path 对象。
    """
    # safe_load 只按 YAML 数据格式解析，不会像执行脚本一样运行配置内容。
    with config_path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    # 逐个处理数据目录和结果目录，统一支持“绝对路径”和“相对仓库路径”。
    for path_key in ("data_dir", "history_data_dir", "output_dir"):
        configured_path = Path(config[path_key]).expanduser()
        # 如果写的是相对路径，就以仓库根目录为起点，避免运行位置不同导致找不到文件。
        if not configured_path.is_absolute():
            configured_path = PROJECT_ROOT / configured_path
        config[path_key] = configured_path

    # 调用者随后可直接使用 config["data_dir"] / "application_train.csv"。
    return config
