import sys
import os
import configparser as cp
import time
work_path = os.path.dirname(os.getcwd())
sys.path.append(work_path)
import numpy as np
import pandas as pd
from IPython import embed
from mapping import *

class mapping_test():
    def __init__(self,mix_tile_path):
        tile_config = cp.ConfigParser()
        tile_config.read(mix_tile_path, encoding='UTF-8')
        self.tile_num=list(map(int,tile_config.get('tile','tile_num').split(',')))
        self.topology=int(tile_config.get('tile','topology'))
        self.c=int(tile_config.get('tile','c'))
        self.tile_connection=int(tile_config.get('tile','tile_connection'))
        self.device_type = [tile_config.get('tile', f'device_type{i}').split(',') for i in range(self.tile_num[0])]
        self.PE_num=[tile_config.get('tile', f'PE_num{i}').split(',') for i in range(self.tile_num[0])]
        for i in range(len(self.PE_num)):
            for j in range(len(self.PE_num[i])):
                self.PE_num[i][j]=int(self.PE_num[i][j])
        self.xbar_size=[tile_config.get('tile', f'xbar_size{i}').split(',') for i in range(self.tile_num[0])]
        for i in range(len(self.xbar_size)):
            for j in range(len(self.xbar_size[i])):
                self.xbar_size[i][j]=int(self.xbar_size[i][j])
        self.PE_group=int(tile_config.get('tile','PE_group'))
        self.layer_mapping=[tile_config.get('tile', f'layer_map_mix{i}').split(',') for i in range(self.tile_num[0])]
        self.auto_layer_mapping=int(tile_config.get('tile',f'auto_layer_mapping'))
        self.layers_sub_loads = []
        self.layers_device_type = []
        self.layers_PE_num = []
        self.layers_xbar_size = []
        self.layertilenum = []
    
    def change_mapping(self,mapping_layout=0):
        self.tile_num=self.tile_num
        max_value = float('-inf')  # 初始化为负无穷大，确保任何数值都比它大
        for row in self.layer_mapping:
            for element in row:
                try:
                    num = int(element)
                    if num > max_value:
                        max_value = num
                except ValueError:
                    continue
        self.layers_sub_loads = [[] for _ in range(max_value+1)]
        self.layers_device_type = [[] for _ in range(max_value+1)]
        self.layers_PE_num = [[] for _ in range(max_value+1)]
        self.layers_xbar_size = [[] for _ in range(max_value+1)]
        for i in range(self.tile_num[0]):
            for j in range(self.tile_num[0]):
                try:
                    layer_num = int(self.layer_mapping[i][j])
                except ValueError:
                    continue
                self.layers_device_type[layer_num].append(self.device_type[i][j])
                self.layers_PE_num[layer_num].append(self.PE_num[i][j])
                self.layers_xbar_size[layer_num].append(self.xbar_size[i][j])
                self.layers_sub_loads[layer_num].append(self.PE_num[i][j]**2*self.xbar_size[i][j]**2)

        for layer in range(max_value+1):
            sorted_indices = sorted(range(len(self.layers_sub_loads[layer])), key=lambda i: self.layers_sub_loads[layer][i], reverse=True)
            self.layers_sub_loads[layer] = [self.layers_sub_loads[layer][i] for i in sorted_indices]
            self.layers_device_type[layer] = [self.layers_device_type[layer][i] for i in sorted_indices]
            self.layers_PE_num[layer] = [self.layers_PE_num[layer][i] for i in sorted_indices]
            self.layers_xbar_size[layer] = [self.layers_xbar_size[layer][i] for i in sorted_indices]
        mapping = map_layers_to_cores(self.tile_num[0], self.layers_sub_loads, mapping_layout=mapping_layout,alpha=1, beta=1, gamma=1, sigma=1)
        output_mapping(self.tile_num[0], mapping, self.layers_sub_loads) 
        self.tile_connection = 4 + mapping_layout
        self.device_type = [['NVM' for _ in range(self.tile_num[0])] for _ in range(self.tile_num[0])]
        self.PE_num = [[2 for _ in range(self.tile_num[0])] for _ in range(self.tile_num[0])]
        self.xbar_size = [[256 for _ in range(self.tile_num[0])] for _ in range(self.tile_num[0])]
        self.layer_mapping = [['no' for _ in range(self.tile_num[0])] for _ in range(self.tile_num[0])]

        for layer in sorted(mapping.keys()):
            sub_loads = self.layers_sub_loads[layer]
            sub_cores = mapping[layer]
            for k in range(len(sub_loads)):
                self.layer_mapping[sub_cores[k][0]][sub_cores[k][1]] = str(layer)
                self.device_type[sub_cores[k][0]][sub_cores[k][1]] = self.layers_device_type[layer][k]
                self.PE_num[sub_cores[k][0]][sub_cores[k][1]] = self.layers_PE_num[layer][k]
                self.xbar_size[sub_cores[k][0]][sub_cores[k][1]] = self.layers_xbar_size[layer][k]

    def read_mapping(self):
        self.tile_num=self.tile_num
        max_value = float('-inf')  # 初始化为负无穷大，确保任何数值都比它大
        for row in self.layer_mapping:
            for element in row:
                try:
                    num = int(element)
                    if num > max_value:
                        max_value = num
                except ValueError:
                    continue
        self.layers_sub_loads = [[] for _ in range(max_value+1)]
        self.layers_device_type = [[] for _ in range(max_value+1)]
        self.layers_PE_num = [[] for _ in range(max_value+1)]
        self.layers_xbar_size = [[] for _ in range(max_value+1)]
        self.layertilenum = [0 for _ in range(max_value+1)]
        if self.topology == 0:
            if self.tile_connection == 0:
                [mapping_order,pos_mapping_order] = self.generate_normal_matrix(self.tile_num[0], self.tile_num[0])
            elif self.tile_connection == 1:
                [mapping_order,pos_mapping_order] = self.generate_snake_matrix(self.tile_num[0], self.tile_num[0])
            elif self.tile_connection == 2:
                [mapping_order,pos_mapping_order] = self.generate_hui_matrix(self.tile_num[0], self.tile_num[0])
            elif self.tile_connection == 3:
                [mapping_order,pos_mapping_order] = self.generate_zigzag_matrix(self.tile_num[0], self.tile_num[0])
            elif self.tile_connection >= 4:
                [mapping_order,pos_mapping_order] = self.generate_zigzag_matrix(self.tile_num[0], self.tile_num[0])
        elif self.topology == 1:
            if self.tile_connection == 0:
                [mapping_order,pos_mapping_order] = self.generate_normal_matrix_cmesh(self.tile_num[0], self.tile_num[0], self.c)
            elif self.tile_connection == 1:
                [mapping_order,pos_mapping_order] = self.generate_snake_matrix_cmesh(self.tile_num[0], self.tile_num[0], self.c)
            elif self.tile_connection == 2:
                [mapping_order,pos_mapping_order] = self.generate_zigzag_matrix_cmesh(self.tile_num[0], self.tile_num[0], self.c)    
            elif self.tile_connection == 3:
                [mapping_order,pos_mapping_order] = self.generate_zigzag_matrix_cmesh(self.tile_num[0], self.tile_num[0], self.c)
            elif self.tile_connection >= 4:
                [mapping_order,pos_mapping_order] = self.generate_zigzag_matrix_cmesh(self.tile_num[0], self.tile_num[0], self.c)
        
        for pos in pos_mapping_order:
            i = int(pos[0])
            j = int(pos[1])
            try:
                layer_num = int(self.layer_mapping[i][j])
            except ValueError:
                continue
            self.layers_device_type[layer_num].append(self.device_type[i][j])
            self.layers_PE_num[layer_num].append(self.PE_num[i][j])
            self.layers_xbar_size[layer_num].append(self.xbar_size[i][j])
            self.layertilenum[layer_num] = self.layertilenum[layer_num] + 1
        return self.layertilenum, self.layers_device_type, self.layers_PE_num, self.layers_xbar_size, self.tile_connection, self.topology, self.c


    def mapping_new_output(self,mix_tile_path_new):
        with open(mix_tile_path_new, 'w') as file:
            file.write(f"[tile]\n")
            file.write(f"tile_num={self.tile_num[0]},{self.tile_num[1]}\n")
            file.write(f"\n")
            for i in range(self.tile_num[0]):
                file.write(f"device_type{i}=")
                for j in range(self.tile_num[0]):
                    if j == self.tile_num[0] - 1 :
                        file.write(f"{self.device_type[i][j]}\n")
                    else :
                        file.write(f"{self.device_type[i][j]},")
            file.write(f"\n")
            for i in range(self.tile_num[0]):
                file.write(f"PE_num{i}=")
                for j in range(self.tile_num[0]):
                    if j == self.tile_num[0] - 1 :
                        file.write(f"{self.PE_num[i][j]}\n")
                    else :
                        file.write(f"{self.PE_num[i][j]},")
            file.write(f"\n")
            file.write(f"PE_group=1\n")
            file.write(f"\n")
            for i in range(self.tile_num[0]):
                file.write(f"xbar_size{i}=")
                for j in range(self.tile_num[0]):
                    if j == self.tile_num[0] - 1 :
                        file.write(f"{self.xbar_size[i][j]}\n")
                    else :
                        file.write(f"{self.xbar_size[i][j]},")
            file.write(f"\n")
            for i in range(self.tile_num[0]):
                file.write(f"layer_map_mix{i}=")
                for j in range(self.tile_num[0]):
                    if j == self.tile_num[0] - 1 :
                        file.write(f"{self.layer_mapping[i][j]}\n")
                    else :
                        file.write(f"{self.layer_mapping[i][j]},")
            file.write(f"\n")
            file.write(f"auto_layer_mapping={self.auto_layer_mapping}\n")
            file.write(f"\n")
            file.write(f"tile_connection={self.tile_connection}\n")
            file.write(f"\n")
            file.write(f"topology={self.topology}\n")
            file.write(f"\n")
            file.write(f"c={self.c}\n")
            #self.update_ini_file('./SimConfig.ini',self.tile_connection)

    def generate_normal_matrix(self, row, column):
        matrix = np.zeros([row, column])
        pos=np.zeros([row*column,2])
        start = 0
        for i in range(row):
            for j in range(column):
                matrix[i][j] = start
                pos[start][0]=i
                pos[start][1]=j
                start += 1
        return matrix,pos

    def generate_snake_matrix(self, row, column):
        matrix = np.zeros([row, column])
        pos=np.zeros([row*column,2])
        start = 0
        for i in range(row):
            for j in range(column):
                if i % 2:
                    matrix[i][column - j - 1] = start
                    pos[start][0]=i
                    pos[start][1]=column - j - 1
                else:
                    matrix[i][j] = start
                    pos[start][0]=i
                    pos[start][1]=j
                start += 1
        return matrix,pos

    def generate_hui_matrix(self, row, column):
        matrix = np.zeros([row, column])
        state = 0
        stride = 1
        step = 0
        start = 0
        dl = 0
        ru = 0
        i = 0
        j = 0
        pos=np.zeros([row*column,2])
        for x in range(row * column):
            if x == 0:
                matrix[i][j] = start
                pos[start][0]=i
                pos[start][1]=j
            else:
                if state == 0:
                    j += 1
                    matrix[i][j] = start
                    pos[start][0]=i
                    pos[start][1]=j
                    state = 1
                elif state == 1:
                    if dl == 0:
                        i += 1
                        matrix[i][j] = start
                        pos[start][0]=i
                        pos[start][1]=j
                        step += 1
                        if step == stride:
                            dl = 1
                            step = 0
                    elif dl == 1:
                        j -= 1
                        matrix[i][j] = start
                        pos[start][0]=i
                        pos[start][1]=j
                        step += 1
                        if step == stride:
                            dl = 0
                            step = 0
                            stride += 1
                            state = 2
                elif state == 2:
                    i += 1
                    matrix[i][j] = start
                    pos[start][0]=i
                    pos[start][1]=j
                    state = 3
                elif state == 3:
                    if ru == 0:
                        j += 1
                        matrix[i][j] = start
                        pos[start][0]=i
                        pos[start][1]=j
                        step += 1
                        if step == stride:
                            ru = 1
                            step = 0
                    elif ru == 1:
                        i -= 1
                        matrix[i][j] = start
                        pos[start][0]=i
                        pos[start][1]=j
                        step += 1
                        if step == stride:
                            ru = 0
                            step = 0
                            stride += 1
                            state = 0
            start += 1
        return matrix,pos

    def generate_zigzag_matrix(self, row, column):
        matrix = np.zeros([row, column])
        pos=np.zeros([row*column,2])
        state = 0
        stride = 1
        step = 0
        i = 0
        j = 0
        start = 0
        for x in range(row * column):
            if x == 0:
                matrix[i][j] = start
            else:
                if state == 0:
                    if j < column - 1:
                        j += 1
                        matrix[i][j] = start
                    else:
                        i += 1
                        matrix[i][j] = start
                    state = 1
                elif state == 1:
                    i += 1
                    j -= 1
                    matrix[i][j] = start
                    step += 1
                    if i == row - 1:
                        state = 2
                        stride -= 1
                        step = 0
                    elif step == stride:
                        state = 2
                        stride += 1
                        step = 0
                elif state == 2:
                    if i < row - 1:
                        i += 1
                        matrix[i][j] = start
                    else:
                        j += 1
                        matrix[i][j] = start
                    state = 3
                elif state == 3:
                    j += 1
                    i -= 1
                    matrix[i][j] = start
                    step += 1
                    if j == column - 1:
                        state = 0
                        stride -= 1
                        step = 0
                    elif step == stride:
                        state = 0
                        stride += 1
                        step = 0
            pos[start][0]=i
            pos[start][1]=j
            start += 1
        return matrix,pos

    def generate_normal_matrix_cmesh(self, row, column, c=2):
        matrix_min,pos_min=self.generate_normal_matrix(c,c)
        matrix = np.zeros([row, column])
        pos=np.zeros([row*column,2])
        start = 0
        for i in range(int(row/c)):
            for j in range(int(column/c)):
                for m in range(c):
                    for n in range(c):
                        matrix[i*c+m][j*c+n] = start*c**2+int(matrix_min[m][n])
                        pos[start*c**2+int(matrix_min[m][n])][0]=i*c+m
                        pos[start*c**2+int(matrix_min[m][n])][1]=j*c+n
                start += 1
        return matrix,pos

    def generate_snake_matrix_cmesh(self, row, column, c=2):
        matrix_min,pos_min=self.generate_snake_matrix(c,c)
        matrix = np.zeros([row, column])
        pos=np.zeros([row*column,2])
        start = 0
        for i in range(int(row/c)):
            for j in range(int(column/c)):
                for m in range(c):
                    for n in range(c):
                        if i % 2:
                            matrix[i*c+m][(int(column/c) - j - 1)*c+n] = start*c**2+int(matrix_min[m][n])
                            pos[start*c**2+int(matrix_min[m][n])][0]=i*c+m
                            pos[start*c**2+int(matrix_min[m][n])][1]=(int(column/c) - j - 1)*c+n
                        else:
                            matrix[i*c+m][j*c+n] = start*c**2+int(matrix_min[m][n])
                            pos[start*c**2+int(matrix_min[m][n])][0]=i*c+m
                            pos[start*c**2+int(matrix_min[m][n])][1]=j*c+n
                start += 1
        return matrix,pos

    def generate_zigzag_matrix_cmesh(self, row, column, c=2):
        matrix_min,pos_min=self.generate_zigzag_matrix(c,c)
        matrix = np.zeros([row, column])
        pos=np.zeros([row*column,2])
        state = 0
        stride = 1
        step = 0
        i = 0
        j = 0
        start = 0
        for x in range(int(row/c) * int(column/c)):
            if x == 0:
                for m in range(c):
                    for n in range(c):
                        matrix[i*c+m][j*c+n] = start*c**2+int(matrix_min[m][n])
                        pos[start*c**2+int(matrix_min[m][n])][0]=i*c+m
                        pos[start*c**2+int(matrix_min[m][n])][1]=j*c+n
            else:
                if state == 0:
                    if j < int(column/c) - 1:
                        j += 1
                        for m in range(c):
                            for n in range(c):
                                matrix[i*c+m][j*c+n] = start*c**2+int(matrix_min[m][n])
                                pos[start*c**2+int(matrix_min[m][n])][0]=i*c+m
                                pos[start*c**2+int(matrix_min[m][n])][1]=j*c+n
                    else:
                        i += 1
                        for m in range(c):
                            for n in range(c):
                                matrix[i*c+m][j*c+n] = start*c**2+int(matrix_min[m][n])
                                pos[start*c**2+int(matrix_min[m][n])][0]=i*c+m
                                pos[start*c**2+int(matrix_min[m][n])][1]=j*c+n
                    state = 1
                elif state == 1:
                    i += 1
                    j -= 1
                    for m in range(c):
                        for n in range(c):
                            matrix[i*c+m][j*c+n] = start*c**2+int(matrix_min[m][n])
                            pos[start*c**2+int(matrix_min[m][n])][0]=i*c+m
                            pos[start*c**2+int(matrix_min[m][n])][1]=j*c+n
                    step += 1
                    if i == int(row/c) - 1:
                        state = 2
                        stride -= 1
                        step = 0
                    elif step == stride:
                        state = 2
                        stride += 1
                        step = 0
                elif state == 2:
                    if i < int(row/c) - 1:
                        i += 1
                        for m in range(c):
                            for n in range(c):
                                matrix[i*c+m][j*c+n] = start*c**2+int(matrix_min[m][n])
                                pos[start*c**2+int(matrix_min[m][n])][0]=i*c+m
                                pos[start*c**2+int(matrix_min[m][n])][1]=j*c+n
                    else:
                        j += 1
                        for m in range(c):
                            for n in range(c):
                                matrix[i*c+m][j*c+n] = start*c**2+int(matrix_min[m][n])
                                pos[start*c**2+int(matrix_min[m][n])][0]=i*c+m
                                pos[start*c**2+int(matrix_min[m][n])][1]=j*c+n
                    state = 3
                elif state == 3:
                    j += 1
                    i -= 1
                    for m in range(c):
                        for n in range(c):
                            matrix[i*c+m][j*c+n] = start*c**2+int(matrix_min[m][n])
                            pos[start*c**2+int(matrix_min[m][n])][0]=i*c+m
                            pos[start*c**2+int(matrix_min[m][n])][1]=j*c+n
                    step += 1
                    if j == int(column/c) - 1:
                        state = 0
                        stride -= 1
                        step = 0
                    elif step == stride:
                        state = 0
                        stride += 1
                        step = 0
            start += 1
        return matrix,pos

    def generate_dynamic_matrix(self, row, column):
        file_name = 'mapping_order.txt'
        pos=np.zeros([row*column,2])
        tile = 0
        data = []
        with open(file_name, 'r') as file:
            for line in file:
                row_data = [int(num) for num in line.split()]
                for num in row_data:
                    pos[num][0]=tile//column
                    pos[num][1]=tile%column
                    tile = tile + 1
                    data.append(num)
        matrix = np.array(data)
        matrix = matrix.reshape((row, column))

        return matrix,pos

start_time = time.time()
MP = mapping_test('mix_tileinfo.ini')
MP.change_mapping(2)
MP.mapping_new_output('mix_tileinfo_new.ini')
endtime = time.time()
print(endtime-start_time)