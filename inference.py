import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # 切换到国内镜像
# Install minimal dependencies (`torch`, `transformers`, `timm`, `tokenizers`, ...)
# > pip install -r https://raw.githubusercontent.com/openvla/openvla/main/requirements-min.txt
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
import torch
import requests  # 用于下载示例图片

# 1. Load Processor & VLA (模型会自动下载或从缓存加载)
processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
vla = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b",
    attn_implementation="flash_attention_2",  # [可选] 需安装flash_attn
    torch_dtype=torch.bfloat16, 
    low_cpu_mem_usage=True, 
    trust_remote_code=True
).to("cuda:0")

# 2. 使用示例图片（无需实际摄像头）
# 下载一张公开的测试图片（厨房场景，适合机器人任务推理）
image_url = "https://images.pexels.com/photos/1640777/pexels-photo-1640777.jpeg"
image = Image.open(requests.get(image_url, stream=True).raw)
# image.show()  # 显示图片（可选）

# 3. 构造测试指令（模拟机器人任务）
prompt = "In: What action should the robot take to pick up the cup?\nOut:"

# 4. 模型推理（验证是否正常运行）
inputs = processor(prompt, image).to("cuda:0", dtype=torch.bfloat16)
with torch.no_grad():
    action = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)

# 5. 打印原始输出（验证模型是否工作）
print("\n=== 模型原始输出 ===")
print(action)
#print(processor.decode(output[0], skip_special_tokens=True))

