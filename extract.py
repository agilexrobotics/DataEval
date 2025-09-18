import os
import argparse
import numpy as np
import torch
import time
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
from tqdm import tqdm

def print_color(text, color='yellow'):
    colors = {'red': '\033[91m', 'green': '\033[92m', 'yellow': '\033[93m', 'blue': '\033[94m', 'end': '\033[0m'}
    print(f"{colors[color]}[LOG] {text}{colors['end']}")

def init_model():
    model_path = "openvla/openvla-7b"
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    ).eval()
    # print_color(f"Model class: {model.__class__}", 'yellow')
    print_color(f"Model loaded on {next(model.parameters()).device}", 'green')
    return processor, model

def extract_view_features(view_path, processor, model, n_segments=3):
    """提取单个视角特征（使用熵权重分阶段聚合）"""
    image_files = sorted([f for f in os.listdir(view_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    if not image_files:
        raise ValueError(f"No images in {view_path}")
    
    # 1. 提取所有帧特征
    frame_features = []
    for img_file in tqdm(image_files, desc=f"Processing {os.path.basename(view_path)}"):
        image = Image.open(os.path.join(view_path, img_file))
        inputs = processor("good", images=image, return_tensors="pt")
        inputs.pixel_values = inputs.pixel_values[:, :3].to(model.device, dtype=torch.bfloat16)

        
        with torch.no_grad():
            vision_outputs = model.vision_backbone.featurizer(inputs.pixel_values)
            # print_color(f"Featurizer output type: {type(vision_outputs)}", 'yellow')
            if hasattr(model.vision_backbone.featurizer, 'get_intermediate_layers'):
                vision_outputs = model.vision_backbone.featurizer.get_intermediate_layers(inputs.pixel_values, n=1)[0]
            frame_feat = vision_outputs[:, 0].float().cpu().numpy().squeeze()
            frame_features.append(frame_feat)
    
    # 2. 分阶段熵权重聚合
    def entropy_weighted_mean(features):
        features = np.clip(features, 1e-10, None)
        entropy = -np.sum(features * np.log(features), axis=1)
        weights = entropy / (np.sum(entropy) + 1e-10)
        return np.sum(features * weights[:, None], axis=0)
    
    segment_len = len(frame_features) // n_segments
    segments = [
        frame_features[i*segment_len : (i+1)*segment_len] 
        for i in range(n_segments)
    ]
    
    view_feature = np.concatenate([
        entropy_weighted_mean(seg) for seg in segments
    ])
    
    print_color(f"Aggregated feature shape: {view_feature.shape}", 'blue')
    return view_feature

def process_episode(episode_path, output_dir, processor, model, n_segments=3):
    """处理单个episode：融合三个视角特征"""
    episode_name = os.path.basename(episode_path)
    print_color(f"\nProcessing {episode_name}", 'yellow')
    
    # 1. 提取三个视角特征
    view_features = {}
    for view in ['front', 'left', 'right']:
        view_path = os.path.join(episode_path, 'camera', 'color', view)
        if not os.path.exists(view_path):
            print_color(f"Missing view: {view}", 'red')
            continue
            
        try:
            start_time = time.time()
            view_feat = extract_view_features(view_path, processor, model, n_segments)
            view_features[view] = view_feat
            print_color(f"{view} feature shape: {view_feat.shape} | Time: {time.time()-start_time:.1f}s", 'green')
        except Exception as e:
            print_color(f"Failed {view}: {str(e)}", 'red')
    
    # 2. 拼接三个视角特征
    if len(view_features) == 3:
        fused_feature = np.concatenate([
            view_features['front'],
            view_features['left'],
            view_features['right']
        ])
        output_path = os.path.join(output_dir, f"{episode_name}_fused.npy")
        np.save(output_path, fused_feature)
        print_color(f"Fused feature saved to {output_path} | Final shape: {fused_feature.shape}", 'blue')
    else:
        print_color(f"Skipped {episode_name} (missing views)", 'red')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True, help="包含aloha文件夹的路径")
    parser.add_argument("--output_dir", default="fused_features", help="输出目录")
    parser.add_argument("--n_segments", type=int, default=3, help="分段聚合数量")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    processor, model = init_model()
    
    # 遍历所有episode
    aloha_dir = os.path.join(args.data_root, "aloha")
    for episode_dir in sorted(os.listdir(aloha_dir)):
        if episode_dir.startswith('episode'):
            process_episode(
                os.path.join(aloha_dir, episode_dir),
                args.output_dir,
                processor,
                model,
                args.n_segments
            )

if __name__ == "__main__":
    main()