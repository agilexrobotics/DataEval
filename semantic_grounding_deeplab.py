import torch
import torchvision.transforms as T
import numpy as np
import cv2
import os
import time


model = torch.hub.load('pytorch/vision', 'deeplabv3_resnet101', pretrained=True)
model.eval()  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def preprocess_image(image_path, target_size=(512, 512)):
    image = cv2.imread(image_path)
    image = cv2.resize(image, target_size)  # 调整图像大小
    image = image[:, :, ::-1]  # BGR 转 RGB
    image = np.copy(image)  
    image = T.ToTensor()(image)  
    image = image.unsqueeze(0)  
    return image


def infer_image(model, image):
    image = image.to(device)
    with torch.no_grad():
        output = model(image)['out'][0]  # 获取模型输出
    output_predictions = torch.argmax(output, dim=0)  # 选择最大概率的类别
    return output_predictions.cpu().numpy()  # 转回 CPU 并转换为 NumPy 数组


def get_colormap():
    # 这里扩充了 COCO 数据集的 21 个类别的颜色映射
    colormap = np.array([
        [0, 0, 0],  # 背景 (0)
        [128, 0, 0],  # 飞机 (1)
        [0, 128, 0],  # 自行车 (2)
        [128, 128, 0],  # 鸟 (3)
        [0, 0, 128],  # 船 (4)
        [128, 0, 128],  # 瓶子 (5)
        [0, 128, 128],  # 公车 (6)
        [128, 128, 128],  # 汽车 (7)
        [0, 0, 0],  # 猫 (8)
        [0, 0, 128],  # 椅子 (9)
        [128, 128, 128],  # 桌子 (10)
        [255, 0, 0],  # 电视 (11)
        # 可以为其他类别添加颜色...
    ])
    return colormap

# 将类别索引映射到颜色
def apply_colormap(mask, colormap):
    # 检查掩码的形状，如果是 1D 数组，则重新塑形为 2D
    if mask.ndim == 1:
        side_length = int(np.sqrt(mask.size)) 
        mask = mask.reshape((side_length, side_length))
    
    print(f"Mask shape: {mask.shape}")

    
    max_index = np.max(mask)
    if max_index >= colormap.shape[0]:
        print(f"Warning: Max index {max_index} exceeds colormap size {colormap.shape[0]}!")
    
    
    color_mask = colormap[mask]  
    return color_mask


def save_mask(original_image, mask, save_path):
    # 获取类别颜色映射
    colormap = get_colormap()
    color_mask = apply_colormap(mask, colormap)

    if color_mask is None:
        print("Error: Color mask is invalid!")
        return
    print(f"Original image shape: {original_image.shape}")
    print(f"Color mask shape: {color_mask.shape}")

    # 确保掩码和原图大小一致
    if original_image.shape[0] != color_mask.shape[0] or original_image.shape[1] != color_mask.shape[1]:
        print(f"Error: Original image size {original_image.shape} and mask size {color_mask.shape} don't match.")
        return
    

    color_mask = cv2.resize(color_mask, (original_image.shape[1], original_image.shape[0]))
    overlay = cv2.addWeighted(original_image, 0.7, color_mask, 0.3, 0)
    cv2.imwrite(save_path, overlay)




def batch_infer(model, image_paths, batch_size=16):
    all_masks = []
    all_images = []
    for i in range(0, len(image_paths), batch_size):
        batch_images = []
        batch_original_images = []
        for image_path in image_paths[i:i+batch_size]:
            image = preprocess_image(image_path)
            batch_images.append(image)
            original_image = cv2.imread(image_path)
            batch_original_images.append(original_image)
        
        batch_images = torch.cat(batch_images, dim=0)  
        # 推理批次
        batch_masks = infer_image(model, batch_images)
        all_masks.append(batch_masks)
        all_images.append(batch_original_images)
    
    return np.concatenate(all_masks, axis=0), np.concatenate(all_images, axis=0)

def process_images(image_dir, output_dir, batch_size=16):
    image_paths = [os.path.join(image_dir, img) for img in os.listdir(image_dir) if img.endswith(".jpg")]
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    start_time = time.time()
    masks, images = batch_infer(model, image_paths, batch_size=batch_size)
    for i, (mask, original_image) in enumerate(zip(masks, images)):
        save_path = os.path.join(output_dir, f"mask_{i}.png")
        save_mask(original_image, mask, save_path)

    end_time = time.time()
    print(f"处理 {len(image_paths)} 张图片用时: {end_time - start_time:.2f} 秒")

if __name__ == "__main__":
    image_dir = "mcap_visualization/data_sample/fold_short_red/aloha/episode1/camera/color/front"  # 替换为实际图像目录
    output_dir = "mcap_visualization/data_sample/fold_short_red/aloha/episode1/camera/color/front_masks"  # 替换为保存分割结果的目录
    batch_size = 16  
    process_images(image_dir, output_dir, batch_size)
