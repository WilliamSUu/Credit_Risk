"""对当前 XGBoost 冠军模型执行固定验证折的 TreeSHAP 全局与客户级解释。"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import StratifiedKFold

from src.bureau_features import CUSTOMER_ID, aggregate_bureau, load_bureau
from src.config import load_config
from src.cross_validate_champion import build_cv_xgboost
from src.data import load_application_data
from src.features import build_preprocessor, make_features
from src.installment_features import build_installment_features
from src.previous_application_features import build_previous_application_features


# 预先设定解释样本量和输出图规模，控制SHAP内存与运行时间。
EXPLANATION_SAMPLE_SIZE = 20_000
BACKGROUND_SAMPLE_SIZE = 1_000
BEESWARM_SAMPLE_SIZE = 10_000
TOP_FEATURE_COUNT = 30
DEPENDENCE_FEATURE_COUNT = 10

# 原始聚合特征名前缀用于把每个模型输入归入对应业务数据源。
SOURCE_PREFIXES = {
    "Application": ("EXT_SOURCE", "AMT_", "DAYS_", "FLAG_", "CNT_", "NAME_", "CODE_", "REG_", "ORGANIZATION_", "OCCUPATION_", "DEF_", "OBS_", "OWN_", "WEEKDAY_", "HOUR_", "LIVE_", "REGION_", "TOTALAREA_", "APARTMENTS_", "BASEMENTAREA_", "COMMONAREA_", "ELEVATORS_", "ENTRANCES_", "FLOORSMAX_", "FLOORSMIN_", "LANDAREA_", "LIVINGAPARTMENTS_", "LIVINGAREA_", "NONLIVINGAPARTMENTS_", "NONLIVINGAREA_", "YEARS_", "EMERGENCYSTATE_", "HOUSETYPE_", "WALLSMATERIAL_", "FONDKAPREMONT_", "HOUR_APPR_", "SELLERPLACE_", "RATE_", "NFLAG_", "PRODUCT_", "CHANNEL_", "SELLER_", "NAME_", "TARGET"),
    "Bureau": ("BUREAU_", "DAYS_CREDIT_", "CREDIT_DAY_", "CREDIT_ACTIVE_", "CREDIT_CURRENCY_", "CREDIT_TYPE_", "AMT_CREDIT_SUM_", "AMT_CREDIT_MAX_", "CNT_CREDIT_", "AMT_ANNUITY_", "HAS_BUREAU"),
    "Previous Application": ("PREV_", "HAS_PREVIOUS_"),
    "Installments": ("INST_", "HAS_INSTALLMENT_"),
}


def identify_source(transformed_name: str) -> str:
    """將 ColumnTransformer 處理後的特徵名對應回原始業務資料來源。"""
    # 移除 numeric__/categorical__ 前綴，再根據原始列名前綴辨認來源。
    raw_name = transformed_name.split("__", 1)[-1]
    # 缺失指示列會使用 missingindicator_原始列名，需要先去掉包裝部分。
    raw_name = raw_name.replace("missingindicator_", "")
    for source, prefixes in SOURCE_PREFIXES.items():
        if raw_name.startswith(prefixes):
            return source
    # application 主表有一些不遵循統一前綴的稀有欄位，默认归入主申请数据。
    return "Application"


def build_champion_tables(config: Dict) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """构建当前冠军特征集，并返回主表标签和客户ID。"""
    application_train, application_test = load_application_data(config["data_dir"])

    # 按既定冠军版本加载三类历史：外部贷款、内部历史申请和真实分期还款。
    bureau = aggregate_bureau(load_bureau(config["history_data_dir"]))
    previous = build_previous_application_features(config["history_data_dir"])
    installments = build_installment_features(config["history_data_dir"])
    champion_history = bureau.merge(previous, on=CUSTOMER_ID, how="outer", validate="one_to_one")
    champion_history = champion_history.merge(
        installments, on=CUSTOMER_ID, how="outer", validate="one_to_one"
    )

    # 仅构造训练标签对应的训练样本；测试数据用于锁定相同字段结构，不参与拟合。
    original_train_rows = len(application_train)
    training = application_train.merge(
        champion_history, on=CUSTOMER_ID, how="left", validate="one_to_one"
    )
    testing = application_test.merge(
        champion_history, on=CUSTOMER_ID, how="left", validate="one_to_one"
    )
    if len(training) != original_train_rows:
        raise RuntimeError("冠军历史特征连接改变了训练集行数。")

    # 所有历史存在标记填0；其它聚合缺失值保留，交由训练折中的预处理器处理。
    for table in (training, testing):
        for column in table.columns:
            if column.startswith("HAS_"):
                table[column] = table[column].fillna(0).astype("int8")

    labels = training["TARGET"].astype("int8")
    customer_ids = training[CUSTOMER_ID].copy()
    features, _, _, numeric_columns, categorical_columns = make_features(training, testing)
    return features, labels, customer_ids


def get_transformed_feature_names(preprocessor) -> List[str]:
    """从已拟合预处理器获取与XGBoost矩阵列顺序一致的真实特征名。"""
    names = preprocessor.get_feature_names_out().tolist()
    # 特征名数量必须等于模型矩阵列数，否则SHAP值无法安全对应回列名。
    if len(names) != len(set(names)):
        raise RuntimeError("预处理后的特征名不唯一，无法可靠地映射SHAP结果。")
    return names


def normalize_shap_values(explainer: shap.TreeExplainer, matrix) -> np.ndarray:
    """返回二分类正类的SHAP矩阵，并兼容不同SHAP版本的输出形状。"""
    values = explainer.shap_values(matrix, check_additivity=False)
    if isinstance(values, list):
        values = values[-1]
    values = np.asarray(values)
    # 新版本可能返回 (样本数, 特征数, 类别数)，取违约正类对应的最后一层。
    if values.ndim == 3:
        values = values[:, :, -1]
    if values.ndim != 2 or values.shape[1] != matrix.shape[1]:
        raise RuntimeError(f"SHAP矩阵形状异常：{values.shape}，模型输入为{matrix.shape}")
    return values


def select_explanation_positions(
    risk: np.ndarray, random_state: int, sample_size: int
) -> np.ndarray:
    """固定随机种子抽取验证客户，保证全局解释可复现。"""
    actual_size = min(sample_size, len(risk))
    generator = np.random.default_rng(random_state)
    # 排序后等距抽取可以覆盖整个风险分布，不会只抽到低风险主流客户。
    ordered = np.argsort(risk)
    selected = np.linspace(0, len(ordered) - 1, actual_size, dtype=int)
    return ordered[selected]


def make_global_tables(
    shap_values: np.ndarray, feature_names: List[str], source_names: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """计算逐特征平均绝对SHAP值和按数据来源汇总的贡献占比。"""
    # 平均绝对值量化该特征对单个客户预测平均影响强度，不表示影响方向。
    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_abs_shap_log_odds": np.abs(shap_values).mean(axis=0),
            "mean_shap_log_odds": shap_values.mean(axis=0),
            "source": source_names,
        }
    ).sort_values("mean_abs_shap_log_odds", ascending=False)
    importance["importance_share"] = (
        importance["mean_abs_shap_log_odds"]
        / importance["mean_abs_shap_log_odds"].sum()
    )

    # 每个来源的特征平均绝对贡献求和，再除以所有特征贡献总量得到相对占比。
    source = (
        importance.groupby("source", as_index=False)["mean_abs_shap_log_odds"]
        .sum()
        .sort_values("mean_abs_shap_log_odds", ascending=False)
    )
    source["contribution_share"] = (
        source["mean_abs_shap_log_odds"] / source["mean_abs_shap_log_odds"].sum()
    )
    return importance, source


def save_global_plots(
    shap_values: np.ndarray,
    display_matrix: pd.DataFrame,
    importance: pd.DataFrame,
    source: pd.DataFrame,
    output_dir: Path,
) -> None:
    """保存全局重要性条形图、方向分布蜂群图和来源贡献图。"""
    top = importance.head(TOP_FEATURE_COUNT).sort_values("mean_abs_shap_log_odds")
    figure, axis = plt.subplots(figsize=(10, 10))
    axis.barh(top["feature"], top["mean_abs_shap_log_odds"], color="#2474B5")
    axis.set(
        title=f"Champion model: top {TOP_FEATURE_COUNT} global SHAP features",
        xlabel="Mean absolute SHAP value (log-odds scale)",
    )
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "shap_global_importance_top30.png", dpi=180)
    plt.close(figure)

    # Beeswarm同时呈现SHAP方向、影响大小和原始特征高低，绘图样本限制为1万名。
    shap.summary_plot(
        shap_values[:BEESWARM_SAMPLE_SIZE],
        display_matrix.iloc[:BEESWARM_SAMPLE_SIZE],
        max_display=TOP_FEATURE_COUNT,
        show=False,
        plot_size=(12, 10),
    )
    plt.title("Champion model SHAP beeswarm: direction and distribution")
    plt.tight_layout()
    plt.savefig(output_dir / "shap_beeswarm_top30.png", dpi=180, bbox_inches="tight")
    plt.close()

    # 来源图展示四组数据各自累计贡献，不将其误解为因果或增量AUC占比。
    source_sorted = source.sort_values("contribution_share")
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.barh(source_sorted["source"], source_sorted["contribution_share"] * 100,
              color=["#E87524", "#36A269", "#D6A32A", "#2474B5"][-len(source_sorted):])
    axis.set(title="SHAP contribution by feature source", xlabel="Share of total mean |SHAP| (%)")
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "shap_source_contribution.png", dpi=180)
    plt.close(figure)


def save_dependence_plots(
    shap_values: np.ndarray,
    raw_display_matrix: pd.DataFrame,
    feature_names: List[str],
    importance: pd.DataFrame,
    output_dir: Path,
) -> pd.DataFrame:
    """为重要业务特征保存依赖图，并登记原始字段与解释特征的对应关系。"""
    rows = []
    selected_features = importance.head(DEPENDENCE_FEATURE_COUNT)["feature"].tolist()
    # SHAP矩阵列顺序来自模型输入，不能误用已经按重要性重新排序后的表格列名。
    shap_frame = pd.DataFrame(shap_values, columns=feature_names)

    for rank, feature in enumerate(selected_features, start=1):
        # 对数值型字段直接绘制原始值；独热字段绘制0/1值并将语义写入图名。
        original_name = feature.split("__", 1)[-1].replace("missingindicator_", "")
        if original_name in raw_display_matrix.columns:
            values = raw_display_matrix[original_name].reset_index(drop=True)
        else:
            # 独热编码列从处理后的显示矩阵读取，若找不到则跳过并记录原因。
            values = None

        safe_name = (
            feature.replace("__", "_").replace("/", "_").replace(" ", "_")
            .replace(":", "_").replace("<", "_").replace(">", "_")
        )[:100]
        if values is None:
            rows.append({"rank": rank, "feature": feature, "plot_file": "", "status": "原始列无法映射，未绘制"})
            continue

        # 缺失填补后的数值用于横轴；缺失指示列本身在横轴上只有0/1。
        values = pd.to_numeric(values, errors="coerce")
        figure, axis = plt.subplots(figsize=(7.5, 5))
        scatter = axis.scatter(
            values.iloc[: len(shap_frame)],
            shap_frame[feature].iloc[: len(values)],
            c=values.iloc[: len(shap_frame)], cmap="coolwarm", s=8, alpha=0.35,
        )
        axis.set(
            title=f"SHAP dependence: {feature}",
            xlabel=f"Feature value: {original_name}",
            ylabel="SHAP value (log-odds; positive raises predicted risk)",
        )
        figure.colorbar(scatter, ax=axis, label="Feature value")
        figure.tight_layout()
        plot_file = f"dependence_{rank:02d}_{safe_name}.png"
        figure.savefig(output_dir / plot_file, dpi=180)
        plt.close(figure)
        rows.append({"rank": rank, "feature": feature, "plot_file": plot_file, "status": "已生成"})
    return pd.DataFrame(rows)


def choose_case_positions(risk: np.ndarray) -> Dict[str, int]:
    """按风险分布选择低风险、中位风险和高风险案例，不依赖真实标签挑样本。"""
    quantiles = {"low_risk": 0.10, "borderline_risk": 0.50, "high_risk": 0.90}
    result = {}
    for name, quantile in quantiles.items():
        target = float(np.quantile(risk, quantile))
        result[name] = int(np.argmin(np.abs(risk - target)))
    return result


def save_customer_explanations(
    explainer: shap.TreeExplainer,
    shap_values: np.ndarray,
    display_matrix: pd.DataFrame,
    probabilities: np.ndarray,
    labels: pd.Series,
    customer_ids: pd.Series,
    selected_positions: np.ndarray,
    output_dir: Path,
) -> pd.DataFrame:
    """输出三名代表客户信息和Waterfall瀑布图，解释高低风险预测构成。"""
    local_positions = choose_case_positions(probabilities[selected_positions])
    rows = []
    feature_names = display_matrix.columns.tolist()

    for case_name, local_position in local_positions.items():
        global_position = int(selected_positions[local_position])
        row_values = shap_values[local_position]
        order = np.argsort(np.abs(row_values))[::-1]
        top_reasons = [
            f"{feature_names[index]} ({row_values[index]:+.3f})"
            for index in order[:5]
        ]
        rows.append(
            {
                "case": case_name,
                "selection_rule": {
                    "low_risk": "抽样验证客户风险分位数约10%",
                    "borderline_risk": "抽样验证客户风险分位数约50%（样本中位附近，不等同业务审批线）",
                    "high_risk": "抽样验证客户风险分位数约90%",
                }[case_name],
                "validation_position": global_position,
                "SK_ID_CURR": int(customer_ids.iloc[global_position]),
                "true_TARGET": int(labels.iloc[global_position]),
                "predicted_default_probability": float(probabilities[global_position]),
                "top_5_absolute_shap_reasons": " | ".join(top_reasons),
            }
        )

        # 使用验证集模型输出的base value和该客户SHAP值绘制局部贡献瀑布图。
        one_row = display_matrix.iloc[[local_position]]
        local_explanation = explainer(one_row, check_additivity=False)
        figure = plt.figure(figsize=(10, 7))
        shap.plots.waterfall(local_explanation[0], max_display=15, show=False)
        plt.title(
            f"{case_name}: predicted risk={probabilities[global_position]:.3%}, "
            f"true TARGET={int(labels.iloc[global_position])}"
        )
        plt.tight_layout()
        figure.savefig(output_dir / f"waterfall_{case_name}.png", dpi=180, bbox_inches="tight")
        plt.close(figure)

    return pd.DataFrame(rows)


def main() -> None:
    """训练冠军模型并在独立验证折上生成完整SHAP解释输出。"""
    config = load_config()
    output_dir = config["output_dir"] / "SHAP analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # 所有历史行为按当前冠军配置聚合为客户级特征，然后通过相同五折生成独立验证集。
    features, labels, customer_ids = build_champion_tables(config)
    splitter = StratifiedKFold(
        n_splits=config["cross_validation_folds"], shuffle=True,
        random_state=config["random_state"],
    )
    folds = list(splitter.split(features, labels))
    train_positions, valid_positions = folds[0]
    x_train = features.iloc[train_positions]
    x_valid = features.iloc[valid_positions]
    y_train = labels.iloc[train_positions]
    y_valid = labels.iloc[valid_positions]

    # 预处理器只在第1折训练集拟合；验证集从未参与缺失值填补和类别字典学习。
    numeric_columns = x_train.select_dtypes(exclude=["object", "category"]).columns.tolist()
    categorical_columns = x_train.select_dtypes(include=["object", "category"]).columns.tolist()
    preprocessor = build_preprocessor(numeric_columns, categorical_columns)
    transformed_train = preprocessor.fit_transform(x_train)
    transformed_valid = preprocessor.transform(x_valid)

    # 树模型采用训练集对应的折号随机种子，树数沿用五折冠军实验中的849棵。
    model = build_cv_xgboost(
        config["random_state"] + 1, config["xgboost_cv_n_estimators"]
    )
    model.fit(transformed_train, y_train)
    all_valid_probabilities = model.predict_proba(transformed_valid)[:, 1]

    # 验证样本按风险排序等距抽样，能覆盖从低到高的客户，而不是仅解释多数低风险样本。
    selected_positions = select_explanation_positions(
        all_valid_probabilities, config["random_state"], EXPLANATION_SAMPLE_SIZE
    )
    explanation_matrix = transformed_valid[selected_positions]

    # TreeSHAP以训练折样本构造背景，解释对象只使用独立验证客户。
    background_generator = np.random.default_rng(config["random_state"])
    background_count = min(BACKGROUND_SAMPLE_SIZE, transformed_train.shape[0])
    background_positions = background_generator.choice(
        transformed_train.shape[0], size=background_count, replace=False
    )
    background_matrix = transformed_train[background_positions]
    # SHAP 0.49 的 interventional TreeExplainer 要求背景数据为稠密数组；
    # 这里只转换1000条抽样背景和2万条解释样本，不把整份训练集转成稠密矩阵。
    if hasattr(background_matrix, "toarray"):
        background_matrix = background_matrix.toarray()
    if hasattr(explanation_matrix, "toarray"):
        explanation_matrix = explanation_matrix.toarray()
    explainer = shap.TreeExplainer(
        model,
        data=background_matrix,
        feature_perturbation="interventional",
        model_output="raw",
    )

    logging.info(
        "计算SHAP：训练背景%s条，验证解释%s条，转换后特征%s个",
        background_count, len(selected_positions), transformed_valid.shape[1],
    )
    shap_values = normalize_shap_values(explainer, explanation_matrix)
    feature_names = get_transformed_feature_names(preprocessor)

    # 保留密集矩阵只用于抽样后的绘图展示；XGBoost训练仍使用原稀疏矩阵节省内存。
    if hasattr(explanation_matrix, "toarray"):
        display_values = explanation_matrix.toarray()
    else:
        display_values = np.asarray(explanation_matrix)
    display_matrix = pd.DataFrame(display_values, columns=feature_names)

    # 计算预处理矩阵中每个列对应的原始字段名称和数据来源，便于按表解释贡献。
    source_names = [identify_source(name) for name in feature_names]
    importance, source_contribution = make_global_tables(
        shap_values, feature_names, source_names
    )
    importance.to_csv(output_dir / "shap_global_importance.csv", index=False)
    source_contribution.to_csv(output_dir / "shap_source_contribution.csv", index=False)
    save_global_plots(shap_values, display_matrix, importance, source_contribution, output_dir)

    # 原始矩阵留给dependence图显示未经缩放的数值；本轮先对SHAP前10项自动生成图。
    raw_selected = x_valid.iloc[selected_positions].reset_index(drop=True)
    dependence_map = save_dependence_plots(
        shap_values, raw_selected, feature_names, importance, output_dir
    )
    dependence_map.to_csv(output_dir / "dependence_plot_index.csv", index=False)

    # SHAP解释和模型分数必须使用同一抽样顺序，客户标签只作案例背景，不参与挑选。
    case_table = save_customer_explanations(
        explainer,
        shap_values,
        display_matrix,
        all_valid_probabilities,
        y_valid.reset_index(drop=True),
        customer_ids.iloc[valid_positions].reset_index(drop=True),
        selected_positions,
        output_dir,
    )
    case_table.to_csv(output_dir / "explained_customers.csv", index=False)

    # 保存实验元信息，确保图表可追溯到具体折、样本量、输出空间和验证指标。
    metadata = {
        "model": "XGBoost champion: Application + Bureau + Previous Application + Installments",
        "validation_fold": 1,
        "cross_validation_folds": config["cross_validation_folds"],
        "training_sample_size": int(len(train_positions)),
        "validation_sample_size": int(len(valid_positions)),
        "shap_explanation_sample_size": int(len(selected_positions)),
        "shap_background_sample_size": int(background_count),
        "transformed_feature_count": int(len(feature_names)),
        "shap_output_space": "raw log-odds; positive SHAP pushes toward default class",
        "shap_version": shap.__version__,
        "xgboost_n_estimators": config["xgboost_cv_n_estimators"],
        "validation_metrics": {
            "roc_auc": float(__import__("sklearn.metrics", fromlist=["roc_auc_score"]).roc_auc_score(y_valid, all_valid_probabilities)),
            "pr_auc": float(__import__("sklearn.metrics", fromlist=["average_precision_score"]).average_precision_score(y_valid, all_valid_probabilities)),
            "ks": float(np.max(
                __import__("sklearn.metrics", fromlist=["roc_curve"]).roc_curve(y_valid, all_valid_probabilities)[1]
                - __import__("sklearn.metrics", fromlist=["roc_curve"]).roc_curve(y_valid, all_valid_probabilities)[0]
            )),
        },
        "source_mapping_note": "由处理后特征名前缀映射；Bureau/Previous/Installments聚合字段显式映射，其余归入Application",
        "customer_case_note": "borderline_risk表示抽样验证客户的中位风险位置，不等同于业务审批阈值",
    }
    with (output_dir / "shap_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)
    logging.info("SHAP解释全部完成，结果位于%s", output_dir)


if __name__ == "__main__":
    main()
