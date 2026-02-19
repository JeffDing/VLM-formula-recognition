#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PNG图片处理工具
功能：
1. 将PNG图片中的透明区域替换为白色
2. 扩展图片画布
"""

import os
import argparse
from PIL import Image


def replace_transparent_with_white(image):
    """
    将图片中的透明区域替换为白色
    
    Args:
        image: PIL Image对象
    
    Returns:
        处理后的PIL Image对象
    """
    # 确保图片有alpha通道
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    
    # 创建一个新的RGB图片，背景为白色
    background = Image.new('RGBA', image.size, (255, 255, 255, 255))
    
    # 将原图合成到白色背景上
    result = Image.alpha_composite(background, image)
    
    return result


def extend_canvas(image, extend_pixels, position='center'):
    """
    扩展图片画布
    
    Args:
        image: PIL Image对象
        extend_pixels: 扩展的像素数（四边均匀扩展）
        position: 原图在新画布中的位置 ('center', 'top-left', 'top', 'top-right', etc.)
    
    Returns:
        扩展画布后的PIL Image对象
    """
    width, height = image.size
    new_width = width + extend_pixels * 2
    new_height = height + extend_pixels * 2
    
    # 创建新的白色背景画布
    new_image = Image.new('RGBA', (new_width, new_height), (255, 255, 255, 255))
    
    # 计算原图在新画布中的位置
    if position == 'center':
        x = extend_pixels
        y = extend_pixels
    elif position == 'top-left':
        x = 0
        y = 0
    elif position == 'top':
        x = extend_pixels
        y = 0
    elif position == 'top-right':
        x = extend_pixels * 2
        y = 0
    elif position == 'left':
        x = 0
        y = extend_pixels
    elif position == 'right':
        x = extend_pixels * 2
        y = extend_pixels
    elif position == 'bottom-left':
        x = 0
        y = extend_pixels * 2
    elif position == 'bottom':
        x = extend_pixels
        y = extend_pixels * 2
    elif position == 'bottom-right':
        x = extend_pixels * 2
        y = extend_pixels * 2
    else:
        # 默认居中
        x = extend_pixels
        y = extend_pixels
    
    # 将原图粘贴到新画布上
    new_image.paste(image, (x, y))
    
    return new_image


def process_images(input_path, output_path, quality=95, extend_pixels=0, position='center'):
    """
    处理指定文件夹下的所有PNG图片
    
    Args:
        input_path: 输入图片文件夹路径
        output_path: 输出图片文件夹路径
        quality: 保存质量 (1-100)
        extend_pixels: 画布扩展像素数
        position: 原图在新画布中的位置
    """
    # 检查输入路径是否存在
    if not os.path.exists(input_path):
        print(f"错误：输入路径 '{input_path}' 不存在！")
        return
    
    # 创建输出目录
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print(f"创建输出目录：{output_path}")
    
    # 获取所有PNG图片
    png_files = [f for f in os.listdir(input_path) if f.lower().endswith('.png')]
    
    if not png_files:
        print(f"在 '{input_path}' 中没有找到PNG图片！")
        return
    
    print(f"找到 {len(png_files)} 个PNG图片，开始处理...")
    print(f"参数设置：质量={quality}%, 画布扩展={extend_pixels}像素, 位置={position}")
    print("-" * 50)
    
    success_count = 0
    error_count = 0
    
    for filename in png_files:
        try:
            input_file = os.path.join(input_path, filename)
            output_file = os.path.join(output_path, filename)
            
            # 打开图片
            image = Image.open(input_file)
            original_mode = image.mode
            
            # 替换透明区域为白色
            image = replace_transparent_with_white(image)
            
            # 扩展画布（如果需要）
            if extend_pixels > 0:
                image = extend_canvas(image, extend_pixels, position)
            
            # 转换为RGB模式保存（因为已经没有透明区域了）
            image = image.convert('RGB')
            
            # 保存图片
            image.save(output_file, 'PNG', quality=quality)
            
            print(f"✓ 处理成功：{filename} (原始模式: {original_mode})")
            success_count += 1
            
        except Exception as e:
            print(f"✗ 处理失败：{filename} - 错误：{str(e)}")
            error_count += 1
    
    print("-" * 50)
    print(f"处理完成！成功：{success_count}，失败：{error_count}")


def main():
    parser = argparse.ArgumentParser(
        description='PNG图片处理工具 - 替换透明区域为白色并扩展画布',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
使用示例：
  python image_processor.py -i ./input -o ./output
  python image_processor.py -i ./input -o ./output -q 90 -e 50
  python image_processor.py -i ./input -o ./output -q 100 -e 100 -p center

位置选项：
  center      - 居中（默认）
  top-left    - 左上角
  top         - 顶部居中
  top-right   - 右上角
  left        - 左侧居中
  right       - 右侧居中
  bottom-left - 左下角
  bottom      - 底部居中
  bottom-right- 右下角
        '''
    )
    
    parser.add_argument('-i', '--input', type=str, required=True,
                        help='输入图片文件夹路径')
    parser.add_argument('-o', '--output', type=str, required=True,
                        help='输出图片文件夹路径')
    parser.add_argument('-q', '--quality', type=int, default=95,
                        help='保存质量 (1-100，默认95)')
    parser.add_argument('-e', '--extend', type=int, default=0,
                        help='画布扩展像素数（默认0，不扩展）')
    parser.add_argument('-p', '--position', type=str, default='center',
                        choices=['center', 'top-left', 'top', 'top-right', 
                                'left', 'right', 'bottom-left', 'bottom', 'bottom-right'],
                        help='原图在新画布中的位置（默认center）')
    
    args = parser.parse_args()
    
    # 验证质量参数
    if args.quality < 1 or args.quality > 100:
        print("错误：质量参数必须在1-100之间！")
        return
    
    # 验证扩展像素参数
    if args.extend < 0:
        print("错误：扩展像素数不能为负数！")
        return
    
    # 处理图片
    process_images(
        input_path=args.input,
        output_path=args.output,
        quality=args.quality,
        extend_pixels=args.extend,
        position=args.position
    )


if __name__ == '__main__':
    main()
