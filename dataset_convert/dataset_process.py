import os
from PIL import Image
from pathlib import Path

def convert_png_to_jpg(input_folder, output_folder=None, quality=95):
    """
    将文件夹中的所有PNG图片转换为JPG格式，透明部分用白色填充
    
    参数:
        input_folder: 输入文件夹路径
        output_folder: 输出文件夹路径（如果为None，则在同一文件夹创建jpg_files子文件夹）
        quality: JPG图片质量，范围1-100，默认95
    """
    # 确保输入文件夹存在
    input_path = Path(input_folder)
    if not input_path.exists():
        print(f"错误：输入文件夹 '{input_folder}' 不存在")
        return
    
    # 设置输出文件夹
    if output_folder is None:
        output_path = input_path / "jpg_files"
    else:
        output_path = Path(output_folder)
    
    # 创建输出文件夹
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 获取所有PNG文件
    png_files = list(input_path.glob("*.png")) + list(input_path.glob("*.PNG"))
    
    if not png_files:
        print(f"在 '{input_folder}' 中没有找到PNG文件")
        return
    
    print(f"找到 {len(png_files)} 个PNG文件")
    print(f"输出文件夹: {output_path}")
    
    converted_count = 0
    skipped_count = 0
    
    for png_file in png_files:
        try:
            # 打开PNG图片
            with Image.open(png_file) as img:
                # 如果图片有透明度通道（RGBA或LA模式）
                if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
                    # 创建一个白色背景
                    white_bg = Image.new('RGB', img.size, (255, 255, 255))
                    
                    # 如果图片是RGBA模式，直接粘贴
                    if img.mode == 'RGBA':
                        white_bg.paste(img, mask=img.split()[3])  # 使用alpha通道作为蒙版
                    # 如果是LA模式（灰度+透明度）
                    elif img.mode == 'LA':
                        # 先转换为RGBA
                        rgba_img = img.convert('RGBA')
                        white_bg.paste(rgba_img, mask=rgba_img.split()[3])
                    # 如果是调色板模式且有透明度
                    elif img.mode == 'P':
                        rgba_img = img.convert('RGBA')
                        white_bg.paste(rgba_img, mask=rgba_img.split()[3])
                    else:
                        # 其他模式直接转换
                        white_bg = img.convert('RGB')
                    
                    result_img = white_bg
                else:
                    # 如果没有透明度，直接转换模式
                    result_img = img.convert('RGB')
                
                # 生成输出文件名（保持原文件名，只改扩展名）
                output_file = output_path / f"{png_file.stem}.jpg"
                
                # 保存为JPG
                result_img.save(output_file, 'JPEG', quality=quality, optimize=True)
                
                converted_count += 1
                print(f"✓ 转换完成: {png_file.name} -> {output_file.name}")
                
        except Exception as e:
            print(f"✗ 转换失败 {png_file.name}: {str(e)}")
            skipped_count += 1
    
    print(f"\n转换完成！")
    print(f"成功转换: {converted_count} 个文件")
    print(f"转换失败: {skipped_count} 个文件")
    print(f"输出文件夹: {output_path}")

def batch_convert_with_options():
    """交互式批量转换函数"""
    print("PNG转JPG转换器")
    print("=" * 40)
    
    # 获取输入文件夹
    while True:
        input_folder = input("请输入PNG图片所在的文件夹路径: ").strip()
        if os.path.exists(input_folder):
            break
        print("文件夹不存在，请重新输入！")
    
    # 选择输出位置
    print("\n选择输出方式:")
    print("1. 在原文件夹创建子文件夹")
    print("2. 指定其他文件夹")
    choice = input("请选择 (1/2): ").strip()
    
    output_folder = None
    if choice == '2':
        output_folder = input("请输入输出文件夹路径: ").strip()
        # 如果文件夹不存在，询问是否创建
        if not os.path.exists(output_folder):
            create = input(f"文件夹不存在，是否创建 '{output_folder}'? (y/n): ").strip().lower()
            if create != 'y':
                print("使用默认输出方式")
                output_folder = None
    
    # 选择图片质量
    while True:
        try:
            quality = int(input("请输入JPG图片质量 (1-100，推荐85-95): ").strip())
            if 1 <= quality <= 100:
                break
            print("请输入1-100之间的数字！")
        except ValueError:
            print("请输入有效的数字！")
    
    # 执行转换
    print("\n开始转换...")
    convert_png_to_jpg(input_folder, output_folder, quality)

if __name__ == "__main__":
    # 方法1: 直接调用函数（修改下面的路径）
    #convert_png_to_jpg("/root/data/dataset/VLM-formula-recognition-dataset-intern-camp/train/mini_train","data/dataset/VLM-formula-recognition-dataset-intern-camp/train/mini_train_new", quality=100)
    
    # 方法2: 使用交互式界面
    batch_convert_with_options()