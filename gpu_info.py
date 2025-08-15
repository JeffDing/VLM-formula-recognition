import os
import time
import csv

def tail_csv(file_path, sleep_interval=1):
    """
    实时读取 CSV 文件并格式化输出指定列
    """
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"文件 '{file_path}' 不存在，请确保文件路径正确！")
        return

    # 打开文件并定位到文件末尾
    file = open(file_path, 'r', newline='', encoding='utf-8')
    file.seek(0, os.SEEK_END)
    
    try:
        print(f"正在监控文件: {file_path}")
        print("按 Ctrl+C 停止程序")

        # 定义感兴趣的列名
        columns_of_interest = ['timestamp', 'utilization.vis_vram.usage [%]', 'utilization.GPU [%]']

        # 读取文件标题行以获取列名
        file.seek(0)
        first_line = file.readline()
        file.seek(0, os.SEEK_END)  # 回到文件末尾

        # 解析标题行以获取列名
        header_reader = csv.reader([first_line])
        header = next(header_reader, [])
        if not header:
            print("CSV 文件没有标题行或标题行为空")
            return

        # 检查是否包含所有感兴趣的列
        if not all(col in header for col in columns_of_interest):
            print(f"CSV 文件缺少某些列: {columns_of_interest}")
            return

        while True:
            # 检查文件是否被修改或追加
            current_position = file.tell()
            line = file.readline()
            
            if line:
                # 使用 csv 读取器解析行
                reader = csv.DictReader([line], fieldnames=header)
                parts = next(reader, None)
                
                if parts is None:
                    print(f"无法解析行: {line.strip()}")
                    continue

                # 检查是否所有感兴趣的列都存在
                if all(col in parts for col in columns_of_interest):
                    try:
                        # 提取需要的列
                        timestamp = parts['timestamp']
                        vram_usage_str = parts['utilization.vis_vram.usage [%]']
                        gpu_usage = parts['utilization.GPU [%]']
                        
                        # 将字符串转换为浮点数
                        vram_usage = float(vram_usage_str)
                        
                        # 格式化输出，保留两位小数并使输出对齐
                        print(f"{timestamp.ljust(30)}显存占用: {vram_usage:.2f}%\t   GPU使用率: {gpu_usage}%")
                    except ValueError as e:
                        print(f"值错误: {e}，无法转换为浮点数。当前行: {line.strip()}")
                        continue
                else:
                    print(f"行中缺少某些列: {line.strip()}")
            else:
                # 如果没有新内容，回到原位置并等待
                file.seek(current_position)
                time.sleep(sleep_interval)
    
    except KeyboardInterrupt:
        print("\n程序已停止")
    except Exception as e:
        print(f"发生错误: {e}")
    finally:
        file.close()

if __name__ == "__main__":
    file_path = "output.csv"  # 默认文件名
    tail_csv(file_path)