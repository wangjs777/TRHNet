# 打开原始文件和目标文件
with open('/22wangjunsong/datasets/Food2k_label/test_finetune.txt', 'r') as infile, open('/22wangjunsong/datasets/Food2k_label/train.txt', 'w') as outfile:
    # 逐行读取原始文件
    for line in infile:
        # 删除每行的最后一个数字（即零），并写入新的文件
        new_line = ' '.join(line.split()[:-1])  # 删除最后一个元素
        outfile.write(new_line + '\n')  # 写入新行，保留换行符