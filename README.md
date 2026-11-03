# Home Credit 信贷违约风险建模与审批策略

这是一个从数据建模延伸到信贷决策的端到端项目。项目使用 Home Credit Default Risk 数据，预测客户发生还款困难的概率，并依次完成历史行为特征工程、分层五折验证、SHAP解释、数据质量治理、概率校准、风险分箱和审批策略模拟。

项目重点不只是提高AUC，而是回答三个更接近真实风控的问题：

1. 历史借贷与还款行为能否稳定提高违约识别能力？
2. 模型依赖哪些信息，是否存在异常值、缺失漂移和概率失真？
3. 如何把预测概率转化为审批率、获批坏账率和误拒率之间的业务权衡？

## 最终结论

正式冠军模型为：

```text
XGBoost
+ Application 主申请信息
+ Bureau 外部征信历史
+ Previous Application 内部历史申请
+ Installments Payments 内部分期还款行为
```

冠军模型使用相同的分层五折交叉验证进行评估：

| 指标 | Application XGBoost基线 | 最终冠军 | 变化 |
|---|---:|---:|---:|
| ROC-AUC ↑ | 0.76166 | **0.78987** | +0.02821 |
| PR-AUC ↑ | 0.24644 | **0.28501** | +0.03857 |
| KS ↑ | 0.39001 | **0.43964** | +0.04963 |
| Brier Score ↓ | 0.06766 | **0.06571** | -0.00196 |

加入Installments后，相对上一版冠军的四项指标均在5/5折胜出。针对性异常治理候选的平均指标略高，但Brier仅2/5折胜出，因此按照保守的模型治理规则保留原冠军。

## 业务策略结果

训练样本共有307,511名客户，其中24,825名`TARGET=1`，全体申请样本的历史违约率为8.07%。冠军模型的OOF风险十分位呈严格单调上升：

- 最低风险10%客户的实际违约率为0.96%。
- 最高风险10%客户的实际违约率为30.56%，是总体的3.79倍。
- 最高风险20%客户集中了57.11%的全部违约客户。
- 在70%模拟审批率下，获批样本坏账率为3.44%，并拦截70.14%的真实违约客户；代价是误拒26.48%的正常客户。

这些数字来自五折OOF策略模拟，不代表模型已经上线后的真实因果收益。由于数据不包含贷款收益、LGD、EAD、资金成本和人工审核成本，本项目不宣称某个阈值是利润最优策略。

## 数据关系

数据文件不提交到Git。主表一行代表一笔当前申请，历史明细表先聚合为“一名客户一行”，再通过`SK_ID_CURR`左连接到主表：

```text
application_train / application_test
    ├── bureau.csv                  外部机构贷款历史
    ├── previous_application.csv   Home Credit内部历史申请
    └── installments_payments.csv  历史分期应还与实还记录
```

`bureau_balance.csv`曾作为候选数据源进行五折实验，但没有带来边际提升，因此没有进入最终冠军特征集。

## 项目结构

```text
credit-risk-baseline/
├── config.example.yaml                  可提交的配置模板
├── requirements.txt                     Python依赖
├── docs/
│   └── PROJECT_RESULTS.md                完整实验路径和结果档案
├── src/
│   ├── train_baseline.py                 最初逻辑回归基线
│   ├── compare_class_weights.py          类别权重实验
│   ├── compare_logistic_xgboost.py       Logistic与XGBoost比较
│   ├── cross_validate_champion.py        分层五折模型验证
│   ├── bureau_features.py                Bureau特征工程
│   ├── previous_application_features.py  Previous Application特征工程
│   ├── installment_features.py           Installments特征工程
│   ├── compare_external_credit_features.py
│   ├── compare_previous_application_features.py
│   ├── compare_installment_features.py
│   ├── analyze_champion_shap.py          SHAP全局与客户级解释
│   ├── analyze_data_quality_governance.py
│   ├── compare_targeted_governance.py    针对性异常治理挑战者
│   └── analyze_credit_decision_strategy.py
└── outputs/                              本地生成结果
```

## 环境与数据配置

建议使用Python 3.9或更高版本。在项目根目录执行：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp config.example.yaml config.yaml
```

然后修改本地`config.yaml`中的两个路径：

- `data_dir`：包含`application_train.csv`和`application_test.csv`。
- `history_data_dir`：包含`bureau.csv`、`previous_application.csv`和`installments_payments.csv`。

`config.yaml`包含个人电脑路径，已被`.gitignore`排除；仓库只保留`config.example.yaml`。

## 推荐复现顺序

所有命令都从项目根目录执行：

```bash
# 1. 最初基线及模型选择
PYTHONPATH=. python src/train_baseline.py
PYTHONPATH=. python src/compare_class_weights.py
PYTHONPATH=. python src/compare_logistic_xgboost.py
PYTHONPATH=. python src/cross_validate_champion.py

# 2. 历史行为特征的增量实验
PYTHONPATH=. python src/compare_external_credit_features.py
PYTHONPATH=. python src/compare_previous_application_features.py
PYTHONPATH=. python src/compare_installment_features.py

# 3. 模型解释、数据治理和业务决策
PYTHONPATH=. python src/analyze_champion_shap.py
PYTHONPATH=. python src/analyze_data_quality_governance.py
PYTHONPATH=. python src/compare_targeted_governance.py
PYTHONPATH=. python src/analyze_credit_decision_strategy.py
```

完整运行会聚合约1360万条Installments付款记录并执行多组五折训练，需要较多时间和内存。审批策略脚本直接复用已保存的OOF预测，不会重新训练模型。

## 方法设计

- 使用分层五折交叉验证，固定随机种子为42。
- 不使用`SK_ID_CURR`作为模型输入。
- 数值缺失使用训练折中位数填补，并保留缺失指示变量。
- 类别变量使用训练折众数填补和独热编码。
- 所有预处理参数、异常截尾线均只从当前训练折学习，避免验证泄漏。
- XGBoost固定使用849棵树；该数量来自先前独立实验的early stopping结果，不借用外层验证折调参。
- 同一轮候选模型共享完全相同的客户折，形成配对比较。
- 同时报告ROC-AUC、PR-AUC、KS与Brier Score，避免只优化单一排序指标。

## 可解释性与模型治理

SHAP结果显示，主申请信息贡献约36.59%，三类历史数据合计约63.40%。重要信号包括外部评分、贷款金额、历史申请到期时间、外部负债利用率以及近期逾期率。

治理分析进一步检查了：

- 小分母导致的负债/授信和实付/应付极端比率；
- 训练集与Kaggle测试集的缺失率漂移；
- 折内截尾前后的五折效果；
- OOF概率校准、ECE、校准截距和斜率；
- 异常治理方案是否具备足够稳定性替换正式冠军。

## 重要限制

- `TARGET`表示数据集定义的早期还款困难，不等同于所有业务中的终身违约。
- 数据没有可直接用于严格时间切分的申请时间，因此五折验证不是Out-of-Time验证。
- Kaggle测试集没有标签，训练/测试漂移只能诊断，不能验证测试集真实效果。
- 历史表缺失经常表示客户没有对应历史，属于结构性缺失，不应全部视为数据错误。
- 审批策略是基于历史标签的OOF模拟，不包含拒绝推断、宏观周期变化或真实上线反馈。
- 人口属性和地区变量可能引起公平性与合规问题；模型性能不能替代公平审查和人工治理。

详细的实验演进、模型取舍和输出文件索引见[项目结果档案](docs/PROJECT_RESULTS.md)。
