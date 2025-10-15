import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

def load_analysis_results(result_file):
    """加载分析结果"""
    df = pd.read_csv(result_file)
    return df

def visualize_cluster_results(df, output_dir):
    """聚类结果可视化"""
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='tsne_x', y='tsne_y', hue='cluster', data=df, palette='viridis', alpha=0.7)
    plt.title('Cluster Visualization')
    plt.xlabel('TSNE Dimension 1')
    plt.ylabel('TSNE Dimension 2')
    plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cluster_visualization.png'))
    plt.close()

def visualize_view_contribution(view_contrib, output_dir):
    """视角贡献度可视化"""
    plt.figure(figsize=(10, 6))
    view_contrib_df = pd.DataFrame.from_dict(view_contrib, orient='index')
    view_contrib_df.plot(kind='bar', stacked=True, legend=True, color=['blue', 'orange', 'green'])
    plt.title('View Contribution per Cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Contribution Ratio')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'view_contribution.png'))
    plt.close()

def visualize_statistics(stats, output_dir):
    """统计信息可视化"""
    # 绘制视角均值
    plt.figure(figsize=(10, 6))
    means = stats["view_means"]
    plt.bar(means.keys(), means.values(), color=['blue', 'orange', 'green'])
    plt.title('View Mean Feature Values')
    plt.ylabel('Mean Value')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'view_means.png'))
    plt.close()

    # 绘制全局均值和标准差
    plt.figure(figsize=(10, 6))
    plt.bar(['Global Mean', 'Global Std'], [stats['global_mean'], stats['global_std']], color='purple')
    plt.title('Global Feature Statistics')
    plt.ylabel('Value')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'global_statistics.png'))
    plt.close()

def visualize_temporal_consistency(temp_consist, output_dir):
    """时序一致性可视化"""
    plt.figure(figsize=(6, 4))
    plt.bar(['Temporal Consistency'], [temp_consist], color='red')
    plt.title('Temporal Consistency')
    plt.ylabel('Consistency Score')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'temporal_consistency.png'))
    plt.close()

def main():
    result_file = 'advanced_analysis/cluster_assignments.csv'  # 分析结果文件
    output_dir = 'visualizations'  # 可视化输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 加载分析结果
    df = load_analysis_results(result_file)

    # 加载视角贡献度信息
    view_contrib = {
        'front': 0.4,
        'left': 0.35,
        'right': 0.25
    }

    # 加载统计信息
    stats = {
        "global_mean": 0.5,
        "global_std": 0.15,
        "view_means": {
            "front": 0.6,
            "left": 0.5,
            "right": 0.45
        }
    }


    temp_consist = 0.85

    # 1. 可视化聚类结果
    visualize_cluster_results(df, output_dir)

    # 2. 可视化视角贡献度
    visualize_view_contribution(view_contrib, output_dir)

    # 3. 可视化统计信息
    visualize_statistics(stats, output_dir)

    # 4. 可视化时序一致性
    visualize_temporal_consistency(temp_consist, output_dir)

    print(f"Visualization complete. Results saved to {output_dir}")

if __name__ == '__main__':
    main()
