import numpy as np
import pandas as pd

def npy_to_txt(npy_file_path, txt_file_path):
    # 加载npy文件
    loaded_array_3d = np.load(npy_file_path, allow_pickle=True)
    
    # 打开txt文件用于写入
    with open(txt_file_path, 'w') as f:
        # 遍历三维数组的每个元素
        for i in range(loaded_array_3d.shape[0]):
            for j in range(loaded_array_3d.shape[1]):
                for k in range(loaded_array_3d.shape[2]):
                    # 将字典转换为字符串格式
                    dict_str = str(loaded_array_3d[i][j][k])
                    # 写入txt文件，每个字典占一行
                    f.write(f"({i}, {j}, {k}): {dict_str}\n")


def npy_to_excel(npy_file_path, excel_file_path):
    # 加载npy文件
    loaded_array_3d = np.load(npy_file_path, allow_pickle=True)
    
    # 创建一个空的DataFrame列表，用于存储每个二维切片的数据
    df_list = []
    
    # 遍历三维数组的每个元素
    for i in range(loaded_array_3d.shape[0]):
        for j in range(loaded_array_3d.shape[1]):
            for k in range(loaded_array_3d.shape[2]):
                # 将字典转换为DataFrame的一行
                df = pd.DataFrame([loaded_array_3d[i][j][k]])
                # 添加三个新列，分别用于标识当前元素的三维索引
                df['i_index'] = i
                df['j_index'] = j
                df['k_index'] = k
                # 将DataFrame添加到列表中
                df_list.append(df)
    
    # 将列表中的所有DataFrame合并为一个大的DataFrame
    merged_df = pd.concat(df_list, ignore_index=True)
    
    # 保存到Excel文件
    merged_df.to_excel(excel_file_path, index=False)

# 调用函数进行转换
npy_file_path = 'area_power.npy'  # npy文件路径
excel_file_path = 'area_power.xlsx'  # 生成的Excel文件路径
npy_to_excel(npy_file_path, excel_file_path)



# # 调用函数进行转换
# npy_file_path = 'area_power.npy'  # npy文件路径
# txt_file_path = 'area_power.txt'  # 生成的txt文件路径
# npy_to_txt(npy_file_path, txt_file_path)