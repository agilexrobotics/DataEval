import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import json
from sklearn.cluster import KMeans, MiniBatchKMeans, SpectralClustering
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
from scipy.stats import kurtosis, skew, f_oneway, zscore
from statsmodels.tsa.stattools import adfuller
import json

def load_features(feature_dir):
    """加载所有融合特征文件"""
    features = {}
    for fname in os.listdir(feature_dir):
        if fname.endswith('_fused.npy'):
            episode = fname.replace('_fused.npy', '')
            features[episode] = np.load(os.path.join(feature_dir, fname))
    return features

def preprocess_features(feature_matrix):
    """特征预处理"""
    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(feature_matrix)
    
    # 视角间归一化
    dim_per_view = feature_matrix.shape[1] // 3
    views = [X_scaled[:, i*dim_per_view:(i+1)*dim_per_view] for i in range(3)]
    views_norm = [v / np.linalg.norm(v, axis=1, keepdims=True) for v in views]
    return np.concatenate(views_norm, axis=1), dim_per_view

def view_contribution_analysis(X, labels, dim_per_view):
    """视角贡献度分析"""
    contributions = []
    for i in range(3):
        view_data = X[:, i*dim_per_view:(i+1)*dim_per_view]
        fvals = []
        for dim in range(view_data.shape[1]):
            groups = [view_data[labels==k, dim] for k in np.unique(labels)]
            if len(groups) > 1:  # 至少两个组才能做ANOVA
                f, _ = f_oneway(*groups)
                fvals.append(f)
        contributions.append(np.mean(fvals) if fvals else 0)
    
    total = sum(contributions)
    return {
        'front': contributions[0]/total,
        'left': contributions[1]/total,
        'right': contributions[2]/total
    }

def temporal_consistency(labels, episodes, fps=30):
    """时序一致性检验"""
    transitions = 0
    valid_episodes = 0
    for ep in np.unique(episodes):
        ep_mask = episodes == ep
        if sum(ep_mask) > fps:  # 至少1秒的数据
            valid_episodes += 1
            ep_labels = labels[ep_mask]
            transitions += np.sum(ep_labels[:-1] != ep_labels[1:])
    
    if valid_episodes == 0:
        return 0
    return 1 - transitions / (sum(ep_mask) - valid_episodes)

def advanced_clustering(X, method='hierarchical'):
    """高级聚类方法"""
    if method == 'hierarchical':
        # 层次化K-Means
        kmeans1 = MiniBatchKMeans(n_clusters=min(50, len(X)//10), batch_size=1000)
        coarse_labels = kmeans1.fit_predict(X)
        
        final_labels = np.zeros_like(coarse_labels)
        for i in np.unique(coarse_labels):
            subgroup = X[coarse_labels == i]
            if len(subgroup) > 10:  # 小样本不继续分割
                kmeans2 = MiniBatchKMeans(n_clusters=max(2, len(subgroup)//20))
                final_labels[coarse_labels == i] = kmeans2.fit_predict(subgroup) + 100*i
            else:
                final_labels[coarse_labels == i] = i
        return final_labels
    
    elif method == 'spectral':
        # 谱聚类
        similarity = np.exp(-squareform(pdist(X))**2 / 0.5)
        spec = SpectralClustering(n_clusters=5, affinity='precomputed')
        return spec.fit_predict(similarity)
    
    else:  # 默认K-Means
        return KMeans(n_clusters=5).fit_predict(X)

def analyze_features(features, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    feature_matrix = np.array(list(features.values()))
    episode_names = np.array(list(features.keys()))
    plt.figure(figsize=(12,4))
    plt.hist(feature_matrix.flatten(), bins=100, log=True)
    plt.title('Feature Value Distribution')
    plt.xlabel('Feature Value')
    plt.ylabel('Count (log)')
    plt.show()
    front = feature_matrix[:, :1024].mean(axis=1)
    left = feature_matrix[:, 1024:2048].mean(axis=1)
    right = feature_matrix[:, 2048:].mean(axis=1)

    a=pd.DataFrame({'Front': front, 'Left': left, 'Right': right}).corr()
    print(a)
    episode_names = np.array(list(features.keys()))
    
    # 1. 数据预处理
    X_normalized, dim_per_view = preprocess_features(feature_matrix)
    
    # 2. 基础统计分析
    stats = {
        "global_mean": np.mean(X_normalized),
        "global_std": np.std(X_normalized),
        "view_means": {
            "front": np.mean(X_normalized[:, :dim_per_view]),
            "left": np.mean(X_normalized[:, dim_per_view:2*dim_per_view]),
            "right": np.mean(X_normalized[:, 2*dim_per_view:])
        }
    }
    
    # 3. 高级聚类
    cluster_labels = advanced_clustering(X_normalized, method='hierarchical')
    unique_clusters = np.unique(cluster_labels)
    
    # 4. 视角贡献度分析
    view_contrib = view_contribution_analysis(X_normalized, cluster_labels, dim_per_view)
    
    # 5. 时序一致性分析
    temp_consist = temporal_consistency(cluster_labels, episode_names)
    
    # 6. 降维可视化
    tsne = TSNE(n_components=2, perplexity=min(30, len(features)-1), random_state=42)
    tsne_results = tsne.fit_transform(X_normalized)
    # 计算每个特征的均值和Z-score
    feature_means = feature_matrix.mean(axis=1)  # 也可以改成 np.linalg.norm(feature_matrix, axis=1)
    zscores = zscore(feature_means)
    
    outlier_mask = np.abs(zscores) > 2
    if np.any(outlier_mask):
        print("\n[Z-score 异常检测结果]")
        for ep, mean_val, z in zip(episode_names[outlier_mask], feature_means[outlier_mask], zscores[outlier_mask]):
            print(f"  Episode: {ep:15s} | Mean={mean_val:.4f} | Z-score={z:.2f}")
    else:
        print("\n[Z-score 异常检测结果] 未发现明显异常 (|z| > 2)")
    # 7. 生成分析报告
    report = {
        "statistics": stats,
        "clustering": {
            "n_clusters": len(unique_clusters),
            "silhouette_score": float(silhouette_score(X_normalized, cluster_labels)),
            "view_contribution": view_contrib,
            "temporal_consistency": temp_consist,
            "cluster_distribution": {str(k):int(v) for k,v in 
                                   zip(*np.unique(cluster_labels, return_counts=True))}
        }
    }
    
    # 8. 保存结果
    pd.DataFrame({
        'episode': episode_names,
        'cluster': cluster_labels,
        'tsne_x': tsne_results[:, 0],
        'tsne_y': tsne_results[:, 1]
    }).to_csv(os.path.join(output_dir, 'cluster_assignments.csv'), index=False)
    
    # with open(os.path.join(output_dir, 'analysis_report.json'), 'w') as f:
    #     json.dump(report, f, indent=2)
    print(report)
    
    # 9. 增强可视化
    plt.figure(figsize=(16, 6))
    
    # 子图1：聚类结果
    plt.subplot(121)
    sns.scatterplot(x=tsne_results[:, 0], y=tsne_results[:, 1], 
                    hue=cluster_labels, palette='viridis', alpha=0.7)
    plt.title(f'Cluster Visualization (Consistency={temp_consist:.2f})')
    
    # 子图2：视角贡献
    plt.subplot(122)
    pd.DataFrame.from_dict(view_contrib, orient='index').plot(
        kind='bar', stacked=True, ax=plt.gca())
    plt.title('View Contribution per Cluster')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'advanced_visualization.png'))
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='机器人操作特征高级分析工具')
    parser.add_argument('--feature_dir', required=True, help='包含_fused.npy的目录')
    parser.add_argument('--output_dir', default='advanced_analysis', help='输出目录')
    args = parser.parse_args()
    
    features = load_features(args.feature_dir)
    print(f"Loaded {len(features)} episode features")
    
    analyze_features(features, args.output_dir)
    print(f"Analysis complete. Results saved to {args.output_dir}")

if __name__ == '__main__':
    main()