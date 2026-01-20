from IPython import embed
import math
from collections import deque

def compute_core_scores_0(N):
    """
    计算每个核心的中心性评分。
    """
    core_scores = {}
    for x in range(N):
        for y in range(N):
            distance_sum = sum(abs(x - i) + abs(y - j) for i in range(N) for j in range(N))
            core_scores[(x, y)] = -distance_sum  # 中心性评分越高越中心
    return core_scores


def compute_core_scores_1(N):
    """
    计算每个核心的中心性评分。
    """
    core_scores = {}
    for x in range(N):
        for y in range(N):
            distance_sum = max(abs(x - (N-1)/2),abs(y - (N-1)/2))
            core_scores[(x, y)] = -distance_sum  # 中心性评分越高越中心
    return core_scores


def compute_core_scores_2(N, gamma=1, sigma=1):
    """
    计算每个核心的综合评分 S(x, y)。
    评分综合考虑中心性和对角线接近性。
    
    参数：
    - N: NoC的维度（N x N）
    - gamma: 中心性评分的权重
    - sigma: 对角线接近性评分的权重
    
    返回：
    - core_scores: 字典，键为核心坐标 (x, y)，值为综合评分 S(x, y)
    """
    core_scores = {}
    center_x, center_y = N // 2, N // 2
    D_max_center = 2 * (N - 1)
    D_max_diag = (N - 1) // 2
    
    for x in range(N):
        for y in range(N):
            # 计算中心性评分
            D_center = abs(x - center_x) + abs(y - center_y)
            S_center = (D_max_center - D_center) / D_max_center  # 归一化到 [0, 1]
            
            # 计算对角线接近性评分
            D_diag = min(abs(x - y), abs(x + y - (N - 1)))
            S_diag = (D_max_diag - D_diag) / D_max_diag if D_max_diag != 0 else 1  # 处理 N=1 的情况
            
            # 综合评分
            S = gamma * S_center + sigma * S_diag
            core_scores[(x, y)] = S
    
    return core_scores

def sort_cores_by_center(core_scores):
    """
    按中心性评分从高到低排序核心。
    """
    sorted_cores = sorted(core_scores.keys(), key=lambda c: core_scores[c], reverse=True)
    return sorted_cores

def sort_layers_by_average_load(layers_sub_loads, gamma=1, sigma=1):
    """
    根据每层的平均子负载和原始层顺序对层进行排序。
    综合考虑平均负载和层顺序，通过加权求和实现平衡。
    
    参数：
    - layers_sub_loads: list of lists, 每个内层列表包含该层的子负载
    - gamma: float, 平均负载的权重
    - sigma: float, 原始层顺序的权重
    
    返回：
    - sorted_layers: list of tuples, 每个元组包含 (层编号, 子负载列表, 平均负载)
                      按综合评分从高到低排序
    """
    layer_info = []
    total_layers = len(layers_sub_loads)
    
    # 计算所有层的平均负载
    average_loads = [sum(sub_loads) / len(sub_loads) if len(sub_loads) > 0 else 0 for sub_loads in layers_sub_loads]
    max_average_load = max(average_loads) if average_loads else 1  # 防止除以零
    
    for l in range(total_layers):
        sub_loads = layers_sub_loads[l]
        average_load = average_loads[l]
        normalized_load = average_load / max_average_load  # 归一化到 [0, 1]
        
        # 计算层顺序评分，越靠前的层评分越高
        normalized_order = 1 - (l / (total_layers - 1)) if total_layers > 1 else 1  # 归一化到 [0, 1]
        
        # 计算综合评分
        combined_score = gamma * normalized_load + sigma * normalized_order
        
        layer_info.append((l, sub_loads, average_load, combined_score))
    
    # 按综合评分从高到低排序
    sorted_layers = sorted(layer_info, key=lambda x: x[3], reverse=True)
    
    # 返回不包含综合评分的排序结果
    sorted_layers = [(l, sl, al) for (l, sl, al, cs) in sorted_layers]
    
    return sorted_layers


def get_neighbors(core, N):
    """
    获取给定核心的所有相邻核心。
    """
    x, y = core
    neighbors = []
    if x > 0:
        neighbors.append((x-1, y))
    if x < N-1:
        neighbors.append((x+1, y))
    if y > 0:
        neighbors.append((x, y-1))
    if y < N-1:
        neighbors.append((x, y+1))
    return neighbors

def find_adjacent_cluster(sorted_cores, required_size, assigned_cores, N, dependency_cores, alpha, beta, core_scores):
    """
    在排序后的核心列表中查找一个尽可能大的相邻核心集群，最多为 required_size。
    综合考虑中心性评分和与依赖层的距离。
    返回集群列表，如果找不到则返回 None。
    """
    for core in sorted_cores:
        if core in assigned_cores:
            continue
        # BFS开始点
        cluster = []
        queue = deque()
        queue.append(core)
        visited = set()
        visited.add(core)
        
        while queue and len(cluster) < required_size:
            current = queue.popleft()
            if current not in assigned_cores and current not in cluster:
                cluster.append(current)
            # 添加相邻核心
            neighbors = get_neighbors(current, N)
            for neighbor in neighbors:
                if neighbor not in visited and neighbor not in assigned_cores:
                    # 计算评分
                    if dependency_cores:
                        D = min(abs(neighbor[0] - dc[0]) + abs(neighbor[1] - dc[1]) for dc in dependency_cores)
                    else:
                        D = 0
                    score = alpha * core_scores[neighbor] - beta * D
                    # 优先添加高评分的核心
                    queue.append(neighbor)
                    visited.add(neighbor)
        if len(cluster) == required_size:
            return cluster
    return None

def map_layers_to_cores(N, layers_sub_loads,mapping_layout=0, alpha=1, beta=1, gamma=1, sigma=1):
    """
    将神经网络层映射到NoC核心上，确保同一层的核心在拓扑上相邻。
    综合考虑核心的中心性评分和与依赖层的距离。
    
    参数：
    - N: NoC的维度（N x N）
    - layers_sub_loads: 每层的子负载列表，列表的每个元素对应一层，元素为该层的子负载列表
    - alpha, beta: 评分函数的权重参数
    
    返回：
    - mapping: 字典，键为层编号，值为映射到的核心列表
    """
    if mapping_layout==0:
        core_scores = compute_core_scores_0(N)
    elif mapping_layout==1:
        core_scores = compute_core_scores_1(N)
    elif mapping_layout>=2:    
        core_scores = compute_core_scores_2(N)
    sorted_cores_initial = sort_cores_by_center(core_scores)
    sorted_layers_info = sort_layers_by_average_load(layers_sub_loads, gamma, sigma)
    
    mapping = {}  # layer -> list of cores
    assigned_cores = set()
    dependency_cores = []  # 记录上一层的核心
    
    for idx, (layer, sub_loads, avg_load) in enumerate(sorted_layers_info):
        required_size = len(sub_loads)
        # 计算每个核心的评分，综合考虑中心性和与依赖层的距离
        sorted_cores = sorted(core_scores.keys(), key=lambda c: (alpha * core_scores[c] - 
                        (beta * min(abs(c[0] - dc[0]) + abs(c[1] - dc[1]) for dc in dependency_cores) 
                        if dependency_cores else 0)), reverse=True)
        
        cluster = find_adjacent_cluster(sorted_cores, required_size, assigned_cores, N, dependency_cores, alpha, beta, core_scores)
        
        if cluster:
            mapping[layer] = cluster
            for core in cluster:
                assigned_cores.add(core)
        else:
            # 如果找不到完整大小的集群，尝试分配尽可能大的集群
            remaining = required_size
            mapping[layer] = []
            size = remaining
            while(remaining > 0):
                cluster = find_adjacent_cluster(sorted_cores, size, assigned_cores, N, dependency_cores, alpha, beta, core_scores)
                if cluster:
                    mapping[layer].extend(cluster)
                    for core in cluster:
                        assigned_cores.add(core)
                    remaining -= size
                    size = remaining
                    if remaining == 0:
                        break
                else:
                    size = size - 1 
            
            if remaining > 0:
                raise Exception(f"无法为层 {layer} 分配所有子负载。")
        
        # 更新依赖层的核心
        dependency_cores = mapping[layer]
    
    return mapping

def display_mapping(mapping, layers_sub_loads):
    """
    显示映射结果。
    """
    for layer in sorted(mapping.keys()):
        sub_loads = layers_sub_loads[layer]
        sub_cores = mapping[layer]
        for k in range(len(sub_loads)):
            print(f"层 {layer} - 子负载 {k+1} (负载: {sub_loads[k]}) 映射到核心 {sub_cores[k]}")

def output_mapping(N, mapping, layers_sub_loads):
    """
    输出映射结果。
    """
    tile_order = 0
    mapping_order = [[-1 for _ in range(N)] for _ in range(N)]
    for layer in sorted(mapping.keys()):
        sub_loads = layers_sub_loads[layer]
        sub_cores = mapping[layer]
        for k in range(len(sub_loads)):
            mapping_order[sub_cores[k][0]][sub_cores[k][1]] = tile_order
            tile_order = tile_order + 1
    for i in range(N):
            for j in range(N):
                if mapping_order[i][j]==-1:
                    mapping_order[i][j] = tile_order
                    tile_order = tile_order + 1
    with open('mapping_order.txt', 'w') as file:
        for row in mapping_order:
            row_str = ' '.join(map(str, row))
            file.write(row_str + '\n')
            

# **示例使用**


## NoC尺寸
#N = 4
#
## 神经网络层的子负载列表
#layers_sub_loads = [
#    [20],            # 层0
#    [30, 60],        # 层1
#    [20, 20],        # 层2
#    [25, 25, 25, 25],# 层3
#    [27, 27, 26]     # 层4
#]
#
## 映射层到核心
#mapping = map_layers_to_cores(N, layers_sub_loads, alpha=1, beta=1)
#
## 显示映射结果
##display_mapping(mapping, layers_sub_loads)
