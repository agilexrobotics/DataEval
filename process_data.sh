#!/bin/bash

# 检查是否提供了输入文件
if [ $# -ne 1 ]; then
    echo "用法: $0 <路径列表文件>"
    echo "文件应包含需要处理的路径列表，每行一个路径"
    exit 1
fi

input_file=$1

# 检查输入文件是否存在
if [ ! -f "$input_file" ]; then
    echo "错误: 文件 $input_file 不存在"
    exit 1
fi

# 读取文件中的每一行路径
while IFS= read -r path || [[ -n "$path" ]]; do
    # 跳过空行
    if [ -z "$path" ]; then
        continue
    fi
    
    echo "正在处理路径: $path"
    
    # 第一步：运行mcap_to_aloha_data.py
    echo "步骤1: 运行mcap_to_aloha_data.py..."
    cd mcap_visualization
    python mcap_to_aloha_data.py --datasetDir "data_sample/$path" --alohaYaml aloha_data_params.yaml
    if [ $? -ne 0 ]; then
        echo "步骤1失败: 处理路径 $path 时出错"
        cd ..
        continue
    fi
    cd ..
    
    # 第二步：运行extract.py
    echo "步骤2: 运行extract.py..."
    python extract.py \
        --data_root "mcap_visualization/data_sample/$path/aloha" \
        --output_dir "features/$path/" \
        --n_segments 3
    if [ $? -ne 0 ]; then
        echo "步骤2失败: 处理路径 $path 时出错"
        continue
    fi
    
    # 第三步：运行embedding_analysis.py
    echo "步骤3: 运行embedding_analysis.py..."
    python embedding_analysis.py --feature_dir "features/$path/" --output_dir "features/$path/"
    if [ $? -ne 0 ]; then
        echo "步骤3失败: 处理路径 $path 时出错"
        continue
    fi
    
    echo "完成处理路径: $path"
    echo "----------------------------------------"
done < "$input_file"

echo "所有路径处理完成"
