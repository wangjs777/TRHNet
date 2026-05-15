import random
from collections import defaultdict

# 创建一个字典用于存储每个类别的图像路径
category_images = defaultdict(list)

# 打开并读取 train.txt 文件
with open('/22wangjunsong/datasets/Food2k_complete/train_balanced.txt', 'r') as infile:
    for line in infile:
        line = line.strip()  # 去除换行符
        category = line.split('/')[0]  # 提取类别名称（文件夹名称）
        category_images[category].append(line)  # 将图像路径按类别存储

# 打开目标文件，写入平衡后的数据
with open('/22wangjunsong/datasets/Food2k_complete/train_balanced.txt', 'w') as outfile:
    for category, images in category_images.items():
        if len(images) > 300:
            # 如果某类别图像数量超过 400，随机采样 400 张
            selected_images = random.sample(images, 300)
        else:
            # 否则保留所有图像
            selected_images = images
        
        # 写入目标文件
        for image in selected_images:
            outfile.write(image + '\n')