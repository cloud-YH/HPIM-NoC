import csv

def get_metrics(row):
    # Row format: Index, Type, PE, Xbar, Conn, TileNum, Area, Power, Latency, Bandwidth
    try:
        area = float(row[6])
        power = float(row[7])
        latency = float(row[8])
        return area, power, latency
    except (ValueError, IndexError):
        return None

def find_best(data, key_func, label):
    best_val = float('inf')
    best_row = None
    best_all_metrics = None
    
    for row in data:
        metrics = get_metrics(row)
        if metrics:
            area, power, latency = metrics
            val = key_func(area, power, latency)
            if val < best_val:
                best_val = val
                best_row = row
                best_all_metrics = (area, power, latency)
    
    if best_row:
        print(f"Minimum {label}:")
        print(f"  Index: {best_row[0]}, Type: {best_row[1]}")
        print(f"  Configuration: PE={best_row[2]}, Xbar={best_row[3]}, Connection={best_row[4]}")
        print(f"  Calculated Value: {best_val:.4e}")
        print(f"  Metrics: Area={best_all_metrics[0]:.4e}, Power={best_all_metrics[1]:.4e}, Latency={best_all_metrics[2]:.4e}")
        print("-" * 30)

file_name = 'Nohetergeneous bak.csv'
data_sram = []
data_nvm = []

try:
    with open(file_name, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row: continue
            # Check for data rows (skip potential headers if any, though file seems headerless)
            if len(row) < 9: continue
            
            try:
                # Validate numeric columns
                float(row[6]) 
            except ValueError:
                continue

            if row[1] == 'SRAM':
                data_sram.append(row)
            elif row[1] == 'NVM':
                data_nvm.append(row)

    print("=== SRAM Analysis ===")
    find_best(data_sram, lambda a, p, l: l, "Latency")
    find_best(data_sram, lambda a, p, l: p * l, "Power * Latency")
    find_best(data_sram, lambda a, p, l: p * l * a, "Power * Latency * Area")

    print("\n=== NVM Analysis ===")
    find_best(data_nvm, lambda a, p, l: l, "Latency")
    find_best(data_nvm, lambda a, p, l: p * l, "Power * Latency")
    find_best(data_nvm, lambda a, p, l: p * l * a, "Power * Latency * Area")

except FileNotFoundError:
    print(f"Error: File {file_name} not found.")