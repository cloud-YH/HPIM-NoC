import torch

def print_dict_structure(data, indent=0):
    """
    递归函数，用于打印字典的结构
    """
    for key, value in data.items():
        print('  ' * indent + str(key) + ':', end=' ')
        if isinstance(value, dict):
            print('dict')
            print_dict_structure(value, indent + 1)
        elif isinstance(value, list):
            print('list')
            if len(value) > 0 and isinstance(value[0], dict):
                print_dict_structure({f'item_{i}': item for i, item in enumerate(value)}, indent + 1)
        elif isinstance(value, torch.Tensor):
            print(f'tensor of shape {tuple(value.shape)}')
        else:
            print(f'{type(value).__name__}')

# 替换为你的.pth文件路径
file_path = '/home/caiangxin/HPIM/MNSIM3.0/cifar10_resnet18_params.pth'

# 加载.pth文件
try:
    data = torch.load(file_path, map_location='cpu')
except FileNotFoundError:
    print(f"Error: File {file_path} not found.")
    exit()
except Exception as e:
    print(f"Error loading file: {e}")
    exit()

# 检查加载的数据是否是字典
if not isinstance(data, dict):
    print(f"The loaded data is not a dictionary. Type: {type(data).__name__}")
else:
    print("Dictionary structure:")
    print_dict_structure(data)