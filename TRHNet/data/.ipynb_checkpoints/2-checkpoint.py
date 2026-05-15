from collections import defaultdict

# 创建一个字典用于存储每个类别的图像数量
category_count = defaultdict(int)

# 打开并读取 train.txt 文件
with open('/22wangjunsong/datasets/Food2k_complete/train_balanced.txt', 'r') as infile:
    for line in infile:
        line = line.strip()  # 去除换行符
        category = line.split('/')[0]  # 提取类别名称（文件夹名称）
        category_count[category] += 1  # 对类别计数

# 将结果保存到新的 txt 文件
with open('./category_counts.txt', 'w') as outfile:
    for category, count in category_count.items():
        outfile.write(f'类别: {category}, 图像数量: {count}\n')

print("结果已保存到 category_counts.txt 文件。")