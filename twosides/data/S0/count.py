import csv

# 输入文件路径
input_file = "valid_ddi.txt"
# 用于存储结果的变量
total_ones = 0
matching_rows = 0

# 读取文件并处理
with open(input_file, 'r') as f:
    # 假设列之间用空格或制表符分隔，根据实际情况调整delimiter
    reader = csv.reader(f, delimiter='\t')  # 若为空格分隔可改为delimiter=' '
    
    for row in reader:
        # 确保每行有四列
        if len(row) != 4:
            continue
            
        # 提取第四列并判断是否为1
        try:
            fourth_col = int(row[3])
            if fourth_col == 1:
                matching_rows += 1
                # 处理第三列的01向量
                vector = row[2].split(',')
                # 统计1的个数
                ones_in_vector = sum(1 for bit in vector if bit == '1')
                total_ones += ones_in_vector
        except (ValueError, IndexError):
            # 跳过格式不正确的行
            continue

print(f"第四列值为1的行共有: {matching_rows} 行")
print(f"这些行中第三列01向量里1的总个数为: {total_ones}")
print(f"平均每行包含1的个数为: {total_ones/matching_rows:.2f}")
    