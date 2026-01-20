import numpy as np
import matplotlib.pyplot as plt
import random
import torch
import time
import math
import copy
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from IPython import embed
import os
import subprocess
import re


class RectBlock:
    def __init__(self, x, y, side, hot, id=0, tag=None, padding=0):
        self.x = x
        self.y = y
        self.side = side
        self.hot = hot
        #self.center_x = x + side / 2
        #self.center_y = y + side / 2
        self.id = id
        self.tag = tag
        self.padding = padding  
        self.layout_side = side + 2 * padding  
        self.center_x = x + self.layout_side / 2
        self.center_y = y + self.layout_side / 2



    def update_position(self, x, y, id=None):
        self.x = x
        self.y = y
        self.id = id if id is not None else self.id
        self.center_x = x + self.layout_side / 2
        self.center_y = y + self.layout_side / 2

    def get_position(self):
        return self.x, self.y

    def get_center(self):
        return self.center_x, self.center_y

def calculate_hpwl(block1, block2):
    x1, y1 = block1.get_center()
    x2, y2 = block2.get_center()
    return 0.5 * (abs(x2 - x1) + abs(y2 - y1))

def total_hpwl(blocks, connections):
    total_length = 0
    path_lengths = {}  # 创建一个字典来存储每条路径的线长

    for i, j in connections:
        length = calculate_hpwl(blocks[i], blocks[j])
        total_length += length
        path_lengths[(i, j)] = length  # 将每条路径的线长存储在字典中
    return total_length, path_lengths

def is_overlapping(block1, block2):
    return not (block1.x + block1.layout_side <= block2.x or
                block1.x >= block2.x + block2.layout_side or
                block1.y + block1.layout_side <= block2.y or
                block1.y >= block2.y + block2.layout_side)

def resolve_overlap(block1, block2):
    overlap_x = min(block1.x + block1.layout_side, block2.x + block2.layout_side) - max(block1.x, block2.x)
    overlap_y = min(block1.y + block1.layout_side, block2.y + block2.layout_side) - max(block1.y, block2.y)

    if overlap_x < overlap_y:
        if block1.center_x < block2.center_x:
            block1.update_position(block1.x - overlap_x / 2, block1.y)
            block2.update_position(block2.x + overlap_x / 2, block2.y)
        else:
            block1.update_position(block1.x + overlap_x / 2, block1.y)
            block2.update_position(block2.x - overlap_x / 2, block2.y)
    else:
        if block1.center_y < block2.center_y:
            block1.update_position(block1.x, block1.y - overlap_y / 2)
            block2.update_position(block2.x, block2.y + overlap_y / 2)
        else:
            block1.update_position(block1.x, block1.y + overlap_y / 2)
            block2.update_position(block2.x, block2.y - overlap_y / 2)

def resolve_all_overlaps(blocks, grid_size):
    """
    解决所有块之间的重叠问题，确保每个块在布局中不重叠。
    :param blocks: 所有块的列表
    """
    # 2. 创建网格进行空间分区
    grid = {}
    for i, block in enumerate(blocks):
        # 计算块所属的网格单元
        cell_x = int(block.x // grid_size)
        cell_y = int(block.y // grid_size)
        grid.setdefault((cell_x, cell_y), []).append(i)

    # 3. 只检查同一网格和相邻网格中的块
    for (cell_x, cell_y), indices in grid.items():
        for i in indices:
            for dx in (-1, 0, 1):  # 相邻网格的x方向偏移
                for dy in (-1, 0, 1):  # 相邻网格的y方向偏移
                    neighbor_cell = (cell_x + dx, cell_y + dy)
                    if neighbor_cell in grid:
                        for j in grid[neighbor_cell]:
                            if i != j and is_overlapping(blocks[i], blocks[j]):
                                resolve_overlap(blocks[i], blocks[j])

def calculate_enclosing_area(blocks):
    """
    计算给定 blocks 布局的外界矩形面积。
    :param blocks: 一个包含所有块的列表，每个块具有 x, y, width, height 属性。
    :return: 外界矩形的面积。
    """
    # 获取所有块的 x 和 y 坐标边界
    min_x = min(block.x for block in blocks)
    max_x = max(block.x + block.side for block in blocks)
    min_y = min(block.y for block in blocks)
    max_y = max(block.y + block.side for block in blocks)

    # 计算外界矩形面积
    return (max_x - min_x) * (max_y - min_y)

def calculate_enclosing_area_1(blocks):
    """
    计算给定 blocks 布局的外界矩形面积。
    :param blocks: 一个包含所有块的列表，每个块具有 x, y, width, height 属性。
    :return: 外界矩形的面积。
    """
    # 获取所有块的 x 和 y 坐标边界
    min_x = min(block.x + block.padding for block in blocks)
    max_x = max(block.x + 2*block.padding + block.side for block in blocks)
    min_y = min(block.y + block.padding for block in blocks)
    max_y = max(block.y + 2*block.padding + block.side for block in blocks)

    # 计算外界矩形面积
    return (max_x - min_x) * (max_y - min_y)

def generate_color(total_colors, color_id, base_colors='tab10', shades_per_base=10):
    """
    生成高对比度颜色，结合离散化基础颜色和深浅变化。

    参数:
        total_colors (int): 需要的总颜色数。
        color_id (int): 当前颜色的索引 (从 0 到 total_colors - 1)。
        base_colors (str): 基础颜色的 colormap 名称（例如 'tab10', 'Set1' 等）。
        shades_per_base (int): 每种基础颜色的深浅变化数。

    返回:
        tuple: (R, G, B, A) 的颜色值。
    """
    if total_colors < 1:
        raise ValueError("total_colors must be at least 1.")
    if not (0 <= color_id < total_colors):
        raise ValueError("color_id must be in the range [0, total_colors - 1].")
    if shades_per_base < 1:
        raise ValueError("shades_per_base must be at least 1.")

    # 获取基础颜色的调色板
    cmap = plt.get_cmap(base_colors)
    base_color_count = cmap.N  # 调色板中的基础颜色数量

    # 每种基础颜色分配的总颜色数
    colors_per_base = (total_colors + base_color_count - 1) // base_color_count
    shades_per_base = min(shades_per_base, colors_per_base)  # 限制深浅变化数

    # 计算当前颜色对应的基础颜色和深浅等级
    base_color_id = color_id // shades_per_base
    shade_id = color_id % shades_per_base

    # 获取基础颜色
    base_color = cmap(int(base_color_id % base_color_count))

    # 调整基础颜色的亮度
    if shades_per_base > 1:
        factor = 1.0 - (shade_id / (shades_per_base - 1)) * 0.5  # 调整亮度，范围为 [1.0, 0.5]
    else:
        factor = 1.0  # 无深浅变化时，直接使用基础颜色

    adjusted_color = tuple(factor * channel for channel in base_color[:3]) + (1.0,)  # 保持透明度为 1.0

    return adjusted_color

def plot_blocks_hot(blocks, connections, layer_num=0, filename="./layout.png"):
   # 生成.flp文件
    all_x = [block.x + block.padding for block in blocks] + [block.x + 2*block.padding + block.side for block in blocks]
    all_y = [block.y + block.padding for block in blocks] + [block.y + 2*block.padding + block.side for block in blocks]
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y) 
    filename="../HotSpot/examples/NoC/ev6.flp"
    with open(filename, "w") as f:
        f.write("# Floorplan close to the Alpha EV6 processor\n")
        f.write("# Line Format: <unit-name>\t<width>\t<height>\t<left-x>\t<bottom-y>\t[<specific-heat>]\t[<resistivity>]\n")
        f.write("# all dimensions are in meters\n")
        f.write("# comment lines begin with a '#'\n")
        f.write("# comments and empty lines are ignored\n")
        f.write("\n")
        i = 1
        for block in blocks:
            if block.side == 0:  # 跳过 side=0 的点
                continue
            f.write("Core{}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\n".format(
                  i,
                  block.side / 100000,
                  block.side / 100000,
                  (block.x + block.padding - min_x) / 100000,
                  (block.y + block.padding - min_y) / 100000
                  ))
            i=i+1

    #生成.ptrace文件
    filename="../HotSpot/examples/NoC/gcc.ptrace"
    with open(filename, "w") as f:
       # 写入表头
       sorted_blocks = sorted((block for block in blocks if block.side != 0), key=lambda x: x.side)
       header = " ".join([f"Core{i+1}" for i, block in enumerate(sorted_blocks)])
       f.write(header + "\n")

       # 写入功耗数据
       for step in range(2):
           line = " ".join(["{:.6f}".format(block.hot) for block in blocks if block.side != 0])
           f.write(line + "\n")

    target_dir = "../HotSpot/examples/NoC"
    script_file = "run.sh"
    original_dir = os.getcwd()

    try:
       os.chdir(target_dir)
       print(f"切换到目录: {target_dir}")

       if not os.path.exists(script_file):
           print(f"错误: 脚本文件 '{script_file}' 不存在")
           exit(1)

       print(f"正在运行脚本: {script_file}")
       result = subprocess.run(["sh", script_file], capture_output=True, text=True, check=True)

       print("脚本输出:")
       print(result.stdout)

       # 检查脚本执行结果
       if result.returncode == 0:
           print("脚本执行成功")
       else:
           print(f"脚本执行失败，退出码: {result.returncode}")

    except subprocess.CalledProcessError as e:
       print(f"脚本执行失败，错误信息: {e}")

    except FileNotFoundError:
       print(f"错误: 目录 '{target_dir}' 不存在")

    finally:
       os.chdir(original_dir)
       print(f"返回到原始目录: {original_dir}")
    
    #max_temp = -1.0  # 初始化为一个极低的温度值
    #max_core = None
#
    #try:
    #    with open('../HotSpot/examples/NoC/outputs/gcc.steady', 'r') as file:
    #        lines = file.readlines()
#
    #        for line in lines:
    #            line = line.strip()  # 移除首尾空白字符
    #            if not line:
    #                continue  # 跳过空行
    #            
    #            # 使用正则表达式判断是否以 "Core" 开头
    #            match = re.match(r'^Core(\d+)', line)
    #            if match:
    #                # 假设每一行的格式是 "CoreX	温度值"，其中 X 是编号，温度值是浮点数
    #                parts = line.split('\t')
    #                if len(parts) >= 2:
    #                    try:
    #                        temp = float(parts[1])
    #                        # 提取核编号
    #                        core_number = int(match.group(1))
#
    #                        if temp > max_temp:
    #                            max_temp = temp
    #                            max_core = core_number  # 更新最高温度对应的核编号
    #                    except ValueError:
    #                        # 如果温度值无法转换为浮点数，跳过该行并打印警告
    #                        print(f"警告: 无法解析温度值: {parts[1]}")
    #                else:
    #                    # 如果分割后的部分不足，跳过该行并打印警告
    #                    print(f"警告: 无法解析行内容: {line}")
    #            else:
    #                # 如果行开头不是 CoreX 格式，跳过该行并打印警告
    #                print(f"警告: 行开头不是 CoreX 格式: {line}")
#
    #    # 输出结果
    #    if max_core is not None:
    #        print(f"Max Temp: {max_temp} K")
    #        print(f"Core: {max_core} core")
    #    else:
    #        print("未找到有效温度数据")
    #except FileNotFoundError:
    #    print(f"错误: 文件gcc.steady未找到")
    #    return None
    #except Exception as e:
    #    print(f"错误: 读取文件时发生异常: {e}")
    #    return None
    max_core_grid = 0
    max_grid_temp = -1.0
    if_central = 0
    # 增加读取 gcc_layer0.grid.steady.txt 文件并找出温度最大值及位置
    grid_file = "../HotSpot/examples/NoC/outputs/gcc_layer0.grid.steady"
    try:
        with open(grid_file, 'r') as file:
            lines = file.readlines()

            max_grid_temp = -1.0  # 初始化为一个极低的温度值
            max_grid_index = -1   # 初始化为-1，表示不在任何核心中
            core_xmin = float('inf')
            core_xmax = -float('inf')
            core_ymin = float('inf')
            core_ymax = -float('inf')

            # 获取所有核心的边界
            for block in blocks:
                if block.side == 0:  # 跳过 side=0 的点
                    continue
                x_min = block.x + block.padding - min_x
                x_max = x_min + block.side + block.padding
                y_min = block.y + block.padding - min_y
                y_max = y_min + block.side + block.padding
                if x_min < core_xmin:
                    core_xmin = x_min
                if x_max > core_xmax:
                    core_xmax = x_max
                if y_min < core_ymin:
                    core_ymin = y_min
                if y_max > core_ymax:
                    core_ymax = y_max

            # 假设文件内容为每行一个温度值，索引从0开始
            for idx, line in enumerate(lines):
                if idx >= 4096:  # 假设网格为64x64，共4096个点
                    break
                try:
                    # 按制表符或空格分割行，提取温度值
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        temp = float(parts[1])  # 假设温度值在第二列
                        if temp > max_grid_temp:
                            max_grid_temp = temp
                            max_grid_index = int(parts[0])  # 假设索引在第一列
                    else:
                        print(f"警告: 行格式不正确，无法解析: {line.strip()}")
                except ValueError:
                    print(f"警告: 无法解析温度值: {line.strip()}")
            # 判断最大温度值的有限元是否在核心区域内
            grid_size = 64  # 假设网格为64x64
            grid_x = max_grid_index % grid_size
            grid_y = max_grid_index // grid_size
            # 中央 4x4 网格的起始和结束索引 (64x64 网格的中央区域)
            central_start = (grid_size // 2) - 2  # 从中心向左和向上偏移 2 个单元
            central_end = (grid_size // 2) + 2    # 从中心向右和向下偏移 2 个单元
            if_central = 0
            if central_start <= grid_x < central_end and central_start <= grid_y < central_end:
                if_central = 1
            grid_x_min = grid_x / grid_size * (core_xmax - core_xmin) + core_xmin
            grid_x_max = (grid_x + 1) / grid_size * (core_xmax - core_xmin) + core_xmin
            grid_y_min = (grid_size - grid_y - 1) / grid_size * (core_ymax - core_ymin) + core_ymin
            grid_y_max = (grid_size - grid_y) / grid_size * (core_ymax - core_ymin) + core_ymin
            in_core = False
            overlap_area = 0.0
            grid_area = (grid_x_max - grid_x_min) * (grid_y_max - grid_y_min)
            half_grid_area = grid_area / 2

            for i, block in enumerate(blocks):
                if block.side == 0:  # 跳过 side=0 的点
                    continue
                x_min = block.x + block.padding - min_x
                x_max = x_min + block.side + block.padding
                y_min = block.y + block.padding - min_y
                y_max = y_min + block.side + block.padding

                # 计算重叠区域
                overlap_x_min = max(grid_x_min, x_min)
                overlap_x_max = min(grid_x_max, x_max)
                overlap_y_min = max(grid_y_min, y_min)
                overlap_y_max = min(grid_y_max, y_max)

                # 如果没有重叠，跳过
                if overlap_x_min >= overlap_x_max or overlap_y_min >= overlap_y_max:
                    continue

                # 计算重叠面积
                overlap_area = (overlap_x_max - overlap_x_min) * (overlap_y_max - overlap_y_min)

                # 判断重叠面积是否超过有限元面积的一半
                if overlap_area > half_grid_area:
                    in_core = True
                    max_core_grid = i+1
                    print(f"最大温度的有限元与核心 {i+1} 的重叠面积超过一半")
                    break

            if not in_core:
                print("最大温度的有限元不在任何核心中，返回-1")
                max_grid_index = -1

            print(f"最大温度值为: {max_grid_temp}，对应的有限元索引: {max_grid_index}")
    except FileNotFoundError:
        print(f"错误: 文件 {grid_file} 未找到")
    except Exception as e:
        print(f"错误: 读取文件时发生异常: {e}")
    #print(f"Max Temp: {max_temp} K")
    #print(f"Core: {max_core} core")
    print(f"Max Temp: {max_grid_temp} K")
    print(f"Core: {max_core_grid} core")
    print(f"If Central: {if_central} core")



def plot_blocks(blocks, connections, layer_num=0, filename="./layout.png"):
    fig, ax = plt.subplots()
    ax.set_aspect('equal')

    # 1. Plot blocks
    for block in blocks:
        color = generate_color(layer_num, block.tag)
        rect = plt.Rectangle((block.x + block.padding, block.y + block.padding), block.side, block.side, fill=True, facecolor=color, edgecolor='black', alpha=0.5)
        ax.add_patch(rect)

    # 2. Plot connections with improved appearance
    for (i, j) in connections:
        x1, y1 = blocks[i].get_center()
        x2, y2 = blocks[j].get_center()
        ax.plot(
            [x1, x2], [y1, y2],
            color='blue', linestyle='-', linewidth=0.6, alpha=0.8
        )

    # 3. Calculate and draw the minimal bounding rectangle
    all_x = [block.x + block.padding for block in blocks] + [block.x + block.padding + block.side for block in blocks]
    all_y = [block.y + block.padding for block in blocks] + [block.y + block.padding + block.side for block in blocks]
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    bounding_rect = plt.Rectangle(
        (min_x, min_y), max_x - min_x, max_y - min_y,
        fill=False, edgecolor='darkgreen', linewidth=2, linestyle='-', label='Bounding Box'
    )
    ax.add_patch(bounding_rect)

    # 4. Set limits to match the bounding rectangle
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)

    # 5. Remove axes and ticks
    ax.axis('off')

    # 6. Save the figure
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

def move_blocks(blocks, spacing_ratio):
    """
    增大块之间的距离，但不改变块的大小。
    :param blocks: 所有块的列表
    :param spacing_ratio: 间距增大的比率（大于1表示增加间距）
    """
    # 计算所有块的中心位置
    centers_x = [block.center_x for block in blocks]
    centers_y = [block.center_y for block in blocks]
    
    # 计算平均中心位置
    avg_center_x = np.mean(centers_x)
    avg_center_y = np.mean(centers_y)
    
    for block in blocks:
        # 计算块与平均中心的向量
        dx = block.center_x - avg_center_x
        dy = block.center_y - avg_center_y
        
        # 根据 spacing_ratio 缩放向量
        scaled_dx = dx * spacing_ratio
        scaled_dy = dy * spacing_ratio
        
        # 更新块的位置
        new_x = avg_center_x + scaled_dx
        new_y = avg_center_y + scaled_dy
        
        # 保持块的中心位置不变，调整位置
        block.update_position(new_x - block.layout_side / 2, new_y - block.layout_side / 2)

def update_positions_and_resolve_overlap(blocks, forces, grid_size=100, axis=None):
    """
    更新块的位置并使用网格分区优化重叠检测。
    :param blocks: 所有块的列表
    :param forces: 每个块的力数组 (N 或 N x 1)
    :param grid_size: 网格的单元大小，用于空间分区
    :param axis: 施加力的轴方向 ('x', 'y', or None)
    """
    # 1. 更新块的位置
    for i, block in enumerate(blocks):
        if axis == 'x':
            new_x = block.x + forces[i]
            block.update_position(new_x, block.y)
        elif axis == 'y':
            new_y = block.y + forces[i]
            block.update_position(block.x, new_y)
        else:
            new_x = block.x + forces[i, 0]
            new_y = block.y + forces[i, 1]
            block.update_position(new_x, new_y)

    # 2. 创建网格进行空间分区并解决重叠
    resolve_all_overlaps(blocks, grid_size)

def improved_smooth_boundary_adjustment(blocks, grid_size, spacing=100, iterations=1):
    """
    改进的边界平滑调整函数，通过成组调整块的位置来减少外接矩形的面积。
    """
    prev_area = calculate_enclosing_area(blocks)
    # print(f"Initial enclosing area: {prev_area}")

    for iteration in range(iterations):
        blocks_old = copy.deepcopy(blocks)  # 深拷贝保存原有的块状态

        # 获取所有块的 x 和 y 坐标边界
        min_x = min(block.x for block in blocks)
        max_x = max(block.x + block.layout_side for block in blocks)
        min_y = min(block.y for block in blocks)
        max_y = max(block.y + block.layout_side for block in blocks)

        # 获取边界块
        left_boundary_blocks = [block for block in blocks if block.x == min_x]
        right_boundary_blocks = [block for block in blocks if block.x + block.layout_side == max_x]
        top_boundary_blocks = [block for block in blocks if block.y + block.layout_side == max_y]
        bottom_boundary_blocks = [block for block in blocks if block.y == min_y]

        # 对每个边界执行内缩操作
        shrink_amount = spacing * 0.05  # 每次迭代的内缩量

        # 左边界内缩
        for block in left_boundary_blocks:
            block.update_position(block.x + shrink_amount, block.y)
        # 右边界内缩
        for block in right_boundary_blocks:
            block.update_position(block.x - shrink_amount, block.y)
        # 顶部边界内缩
        for block in top_boundary_blocks:
            block.update_position(block.x, block.y - shrink_amount)
        # 底部边界内缩
        for block in bottom_boundary_blocks:
            block.update_position(block.x, block.y + shrink_amount)

        # 解决重叠问题
        resolve_all_overlaps(blocks, grid_size=spacing)

        # 计算新的外接矩形面积
        new_area = calculate_enclosing_area(blocks)
        # print(f"Iteration {iteration + 1}, new enclosing area: {new_area}")

        # 如果面积不再减少，则停止迭代
        if new_area >= prev_area:
            blocks = blocks_old  # 恢复到上一次的状态
            break

        prev_area = new_area

def improved_force_directed_layout(blocks, connections, grid_size, topology="mesh", spacing=100, iterations=1000, initial_learning_rate=0.1, tolerance=1):
    """
    改进的力导向布局算法，增加边界平滑和均匀化调整。
    """
    no_change_counter = 0
    prev_hpwl = None

    for iteration in range(iterations):
        learning_rate = initial_learning_rate * (1 - iteration / iterations)

        # ---- 第一步：计算并应用 x 方向的力 ----
        block_centers_x = [block.get_center()[0] for block in blocks]
        forces_x = np.zeros(len(blocks), dtype=np.float32)

        for (i, j) in connections:
            x1, x2 = block_centers_x[i], block_centers_x[j]
            force_x = (x2 - x1) * learning_rate
            forces_x[i] += force_x
            forces_x[j] -= force_x

        # 更新 x 位置并解决重叠
        update_positions_and_resolve_overlap(blocks, forces_x, grid_size=spacing, axis='x')

        # ---- 第二步：计算并应用 y 方向的力 ----
        block_centers_y = [block.get_center()[1] for block in blocks]
        forces_y = np.zeros(len(blocks), dtype=np.float32)

        for (i, j) in connections:
            y1, y2 = block_centers_y[i], block_centers_y[j]
            force_y = (y2 - y1) * learning_rate
            forces_y[i] += force_y
            forces_y[j] -= force_y

        # 更新 y 位置并解决重叠
        update_positions_and_resolve_overlap(blocks, forces_y, grid_size=spacing, axis='y')

        # ---- 第三步：进行边界平滑调整 ----
        improved_smooth_boundary_adjustment(blocks, grid_size, spacing=spacing, iterations=1)

        # 再次解决重叠问题
        resolve_all_overlaps(blocks, grid_size=spacing)

        # ---- 第四步：检查收敛条件 ----
        current_hpwl, _ = total_hpwl(blocks, connections)

        if prev_hpwl is not None:
            hpwl_change = abs(prev_hpwl - current_hpwl)
            if hpwl_change < tolerance:
                no_change_counter += 1
                if no_change_counter >= 10:  # 如果连续10次变化都很小，则认为收敛
                    print(f"Converged at iteration {iteration}")
                    break
            else:
                no_change_counter = 0  # 如果有明显变化，重置计数器
        prev_hpwl = current_hpwl

        # 可选：打印迭代信息
        if iteration % 100 == 0 or iteration == iterations - 1:
            area = calculate_enclosing_area(blocks)
            print(f"Iteration {iteration}, Total HPWL: {current_hpwl}, Enclosing Area: {area}")
        
        #if iteration == 500:
        #    return

def expand_blocks_to_boundary(grid_size, blocks, connections, topology):
    """
    将所有块扩散到外接矩形的边界，使得边上的块与外接矩形相切。
    """
    # 计算当前外接矩形的边界
    original_min_x = min(block.x for block in blocks)
    original_max_x = max(block.x + block.layout_side for block in blocks)
    original_min_y = min(block.y for block in blocks)
    original_max_y = max(block.y + block.layout_side for block in blocks)

    # 计算所有块的中心位置
    centers_x = [block.center_x for block in blocks]
    centers_y = [block.center_y for block in blocks]
    
    # 计算平均中心位置
    avg_center_x = np.mean(centers_x)
    avg_center_y = np.mean(centers_y)

    if topology == "mesh":
        expand_x_min = []
        expand_y_min = []
        expand_x_max = []
        expand_y_max = []
        
        for i in range(grid_size):
            min_x = blocks[i*grid_size].x 
            min_x_id = i*grid_size
            for j in range(grid_size): 
                id = i*grid_size + j
                if blocks[id].x < min_x:
                    min_x = blocks[id].x
                    min_x_id = id
            id = min_x_id
            expand_x_min.append((avg_center_x-original_min_x-blocks[id].layout_side/2)/(avg_center_x-blocks[id].center_x))
            
        for j in range(grid_size):
            min_y = blocks[j].x 
            min_y_id = j
            for i in range(grid_size): 
                id = i*grid_size + j
                if blocks[id].y < min_y:
                    min_y = blocks[id].y
                    min_y_id = id
            id = min_y_id
            expand_y_min.append((avg_center_y-original_min_y-blocks[id].layout_side/2)/(avg_center_y-blocks[id].center_y))

        for i in range(grid_size):
            max_x = blocks[i*grid_size + grid_size - 1].x + blocks[i*grid_size + grid_size - 1].layout_side
            max_x_id = i*grid_size + grid_size - 1
            for j in range(grid_size): 
                id = i*grid_size + j
                if blocks[id].x + blocks[id].layout_side > max_x:
                    max_x = blocks[id].x + blocks[id].layout_side
                    max_x_id = id
            id = max_x_id
            expand_x_max.append((original_max_x-blocks[id].layout_side/2-avg_center_x)/(blocks[id].center_x-avg_center_x))
            
        for j in range(grid_size):
            max_y = blocks[(grid_size - 1)*grid_size + j].x + blocks[(grid_size - 1)*grid_size + j].layout_side
            max_y_id = (grid_size - 1)*grid_size + j
            for i in range(grid_size): 
                id = i*grid_size + j
                if blocks[id].y + blocks[id].layout_side > max_y:
                    max_y = blocks[id].y + blocks[id].layout_side
                    max_y_id = id
            id = max_y_id
            expand_y_max.append((original_max_y-blocks[id].layout_side/2-avg_center_y)/(blocks[id].center_y-avg_center_y))

        for i in range(grid_size):
            for j in range(grid_size):
                id = i*grid_size + j
                if (blocks[id].center_x <= avg_center_x):
                    new_x = avg_center_x - (avg_center_x-blocks[id].center_x)*expand_x_min[i] - blocks[id].layout_side/2
                else:
                    new_x = avg_center_x + (blocks[id].center_x-avg_center_x)*expand_x_max[i] - blocks[id].layout_side/2
                if (blocks[id].center_y <= avg_center_y):
                    new_y = avg_center_y - (avg_center_y-blocks[id].center_y)*expand_y_min[j] - blocks[id].layout_side/2
                else:
                    new_y = avg_center_y + (blocks[id].center_y-avg_center_y)*expand_y_max[j] - blocks[id].layout_side/2
                blocks[id].update_position(new_x, new_y)

    elif topology == "cmesh":
        spacing = 2 * spacing
                
    # 解决移动后的重叠问题
    resolve_all_overlaps(blocks, grid_size=100)

def floorplan(grid_size, topology, area, hot, tag, layer_num=0, iterations=2000, tolerance=1e-4, spacing_ratio=1,  padding_ratio=0.05):
    '''
    grid_size: num of topology nodes for one side
    topology: mesh or cmesh
    area: array for nodes
    tag: layer name for each node
    layer_num: total layer num
    iterations: num of iterations
    tolerance: for early termination
    '''
    blocks = []
    sides = np.sqrt(area)
    
    spacing = 1.5 * np.max(sides)
    padding = (padding_ratio * sides).astype(np.float32)

    if topology == "mesh":
        for i in range(grid_size):
            for j in range(grid_size):
                x = j * spacing
                y = i * spacing
                id = i*grid_size + j
                blocks.append(RectBlock(x, y, sides[i * grid_size + j], hot[i * grid_size + j], id=id, tag=tag[i * grid_size + j], padding=padding[i * grid_size + j]))

        # Define 2D mesh connections between blocks
        connections = []
        for i in range(grid_size):
            for j in range(grid_size):
                idx = i * grid_size + j
                if j < grid_size - 1:  # Connect to right neighbor
                    connections.append((idx, idx + 1))
                if i < grid_size - 1:  # Connect to bottom neighbor
                    connections.append((idx, idx + grid_size))
    elif topology == "cmesh":
        spacing = 2 * spacing
        # Define regular blocks
        for i in range(grid_size):
            for j in range(grid_size):
                x = j * spacing
                y = i * spacing
                id = i*grid_size + j
                blocks.append(RectBlock(x, y, sides[i * grid_size + j], hot[i * grid_size + j], id=id, tag=tag[i * grid_size + j], padding=padding[i * grid_size + j]))
        
        # Insert IO nodes in the center of quadrants
        io_start_index = len(blocks)  # Index where IO blocks start
        for i in range(grid_size // 2):
            for j in range(grid_size // 2):
                x = (2 * j + 0.5) * spacing
                y = (2 * i + 0.5) * spacing
                id = grid_size*grid_size + i*(grid_size//2) + j
                blocks.append(RectBlock(x, y, sides[grid_size * grid_size + i * (grid_size // 2) + j], hot[grid_size * grid_size + i * (grid_size // 2) + j], id=id, tag=tag[grid_size * grid_size + i * (grid_size // 2) + j]))

        # Define Cmesh connections between blocks and IO nodes
        connections = []
        # Connect IO nodes to their neighboring blocks and other IO nodes
        io_count = (grid_size // 2) * (grid_size // 2)
        for idx in range(io_start_index, io_start_index + io_count):
            # Determine the position of the IO node in the grid
            local_idx = idx - io_start_index
            i = local_idx // (grid_size // 2)
            j = local_idx % (grid_size // 2)

            # Connect each IO to the surrounding regular blocks
            connections.append((idx, 2 * i * grid_size + 2 * j))  # Connect to C0
            connections.append((idx, 2 * i * grid_size + 2 * j + 1))  # Connect to C1
            connections.append((idx, 2 * i * grid_size + 2 * j + grid_size))  # Connect to C2
            connections.append((idx, 2 * i * grid_size + 2 * j + grid_size + 1))  # Connect to C3

        # Connect IO nodes to other IO nodes as per the mesh structure
        for i in range(grid_size // 2):
            for j in range(grid_size // 2):
                idx = io_start_index + i * (grid_size // 2) + j
                # Connect to right IO neighbor if it exists
                if j < (grid_size // 2) - 1:
                    connections.append((idx, idx + 1))
                # Connect to bottom IO neighbor if it exists
                if i < (grid_size // 2) - 1:
                    connections.append((idx, idx + (grid_size // 2)))
    

    # Plot the resulting block layout
    #filename = f"./forcedir_{topology}{grid_size}.svg"
    filename = "./forcedir.svg"
    plot_blocks(blocks, connections, layer_num=layer_num, filename=filename)

    initial_hpwl, _ = total_hpwl(blocks, connections)
    initial_area_enclosing_rectangle = calculate_enclosing_area_1(blocks)
    
    print(f'Initial Total HPWL: {initial_hpwl}')
    print(f"Initial Total Area: {initial_area_enclosing_rectangle}")
    
    improved_force_directed_layout(blocks, connections, grid_size, topology=topology, spacing=spacing, iterations=iterations, tolerance=tolerance)

    #expand_blocks_to_boundary(grid_size, blocks, connections, topology)
    move_blocks(blocks, spacing_ratio)
    opt_hpwl, opt_path_lengths = total_hpwl(blocks, connections)
    Area_enclosing_rectangle = calculate_enclosing_area_1(blocks)
    util = sum(area)/Area_enclosing_rectangle
    print(f"Area utilization: {util}")
    print(f'Final Total HPWL: {opt_hpwl}')
    print(f'Area_enclosing_rectangle: {Area_enclosing_rectangle}')

    # Plot the resulting block layout
    #filename = f"./forcedir_opt_{topology}{grid_size}.svg"
    filename = "./forcedir_opt.svg"
    plot_blocks(blocks, connections, layer_num, filename)
    filename = f"./forcedir_opt_{topology}{grid_size}_hot.svg"
    plot_blocks_hot(blocks, connections, layer_num, filename)

    return Area_enclosing_rectangle, opt_hpwl, opt_path_lengths
