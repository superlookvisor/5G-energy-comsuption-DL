# 5G-energy-comsuption-DL

## 5G 基站能耗预测项目 (PI-ANP 版)

本项目旨在利用物理知识引导的注意力神经过程（Physics-Informed Attentive Neural Process, PI-ANP）对 5G 基站的能耗进行精准建模与预测。

### 项目核心模块
- **01_preprocess.py**: 数据清洗与基站级物理指标聚合。
- **02_feature_engineering.py**: 构建物理代理特征（静态、动态及节能代理）。
- **03_physics_baseline.py**: 建立并拟合物理基准模型，提取残差。
- **04_pi_anp_model.py**: 定义 PI-ANP 模型架构，实现物理纠偏。
- **05_pi_anp_evaluation.py**: 模型评估与结果可视化。

### 环境要求
- Python 3.8+
- PyTorch
- NumPy, Pandas, Sklearn, Scipy

### 同步说明
本项目通过 Git 进行版本控制。
- 核心代码：已同步至 GitHub。
- 大型数据：已通过 `.gitignore` 忽略，请手动同步数据文件夹。
