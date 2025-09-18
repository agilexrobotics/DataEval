# Robot Data Feature Extraction & Analysis

## 项目简介
本项目用于机器人操作数据的特征提取和统计分析。通过多视角图像序列，使用 VLA 模型提取视觉特征，并对提取的特征进行高级统计和聚类分析，以评估不同视角的贡献、时序一致性和特征分布特性。

## 功能特性
- **多视角特征提取**：支持前、左、右三个摄像头视角。
- **熵权重分阶段聚合**：对每个视角序列帧特征进行加权聚合。
- **特征融合**：将三个视角的特征拼接保存为统一向量。
- **高级特征分析**：
  - 特征统计（均值、标准差）
  - 高级聚类（层次 K-Means、谱聚类）
  - 视角贡献度评估
  - 时序一致性分析
  - t-SNE 可视化和聚类分布图

## 文件结构
.
├── extract.py # 多视角特征提取
├── embedding_analysis.py # 融合特征分析与可视化
├── fused_features/ # 提取后的融合特征保存目录
└── advanced_analysis/ # 特征分析结果保存目录


## 安装依赖
建议使用 Python >= 3.10 并创建虚拟环境：
```bash
pip install torch torchvision transformers tqdm pillow numpy pandas matplotlib seaborn scikit-learn scipy statsmodels
使用方法
1. 特征提取
从 aloha 数据目录提取多视角特征：

python extract.py --data_root /path/to/data_root --output_dir fused_features --n_segments 3
--data_root：包含 aloha 文件夹的路径

--output_dir：融合特征保存目录（默认 fused_features）

--n_segments：分段聚合数量（默认 3）

输出为每个 episode 的 _fused.npy 文件。

2. 特征分析
对提取的融合特征进行高级分析：


python embedding_analysis.py --feature_dir fused_features --output_dir advanced_analysis
--feature_dir：融合特征文件目录

--output_dir：分析结果保存目录（默认 advanced_analysis）

输出内容：

cluster_assignments.csv：每个 episode 的聚类标签及 t-SNE 坐标

advanced_visualization.png：聚类结果可视化图

控制台打印统计分析报告（均值、标准差、视角贡献度、时序一致性等）

注意事项
数据目录需包含 aloha/episode* 子目录，每个 episode 下包含三个视角图像：


episode1/
└── camera/color/{front,left,right}/*.png
特征提取依赖 openvla/openvla-7b 模型，需联网下载权重。