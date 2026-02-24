import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import random
import os
from typing import Union, Tuple, List, Optional
import math
import glob
from pathlib import Path
import argparse
import multiprocessing
from tqdm import tqdm

class FormulaScanAugmentation:
    """
    完整的公式图像扫描效果数据增强类
    模拟各种扫描仪和打印机的噪声和伪影
    """
    
    def __init__(self, 
                 apply_prob: float = 0.7,
                 noise_level: str = 'medium',
                 scan_type: str = 'mixed'):
        """
        初始化增强器
        
        Args:
            apply_prob: 应用增强的概率 (0-1)
            noise_level: 噪声级别 ['light', 'medium', 'heavy']
            scan_type: 扫描类型 ['document', 'book', 'old_paper', 'mixed']
        """
        self.apply_prob = apply_prob
        self.noise_level = noise_level
        self.scan_type = scan_type
        
        # 根据噪声级别设置参数
        self.noise_params = {
            'light': {
                'gaussian_sigma': (5, 15),
                'salt_pepper_prob': (0.001, 0.005),
                'speckle_sigma': (0.03, 0.08),
                'blur_kernel': (1, 2),
                'brightness_range': (-10, 10),
                'contrast_range': (0.9, 1.1)
            },
            'medium': {
                'gaussian_sigma': (10, 25),
                'salt_pepper_prob': (0.003, 0.01),
                'speckle_sigma': (0.05, 0.12),
                'blur_kernel': (1, 3),
                'brightness_range': (-20, 20),
                'contrast_range': (0.8, 1.2)
            },
            'heavy': {
                'gaussian_sigma': (20, 40),
                'salt_pepper_prob': (0.005, 0.02),
                'speckle_sigma': (0.08, 0.2),
                'blur_kernel': (2, 5),
                'brightness_range': (-30, 30),
                'contrast_range': (0.7, 1.3)
            }
        }
    
    def _get_params(self) -> dict:
        """获取当前噪声级别的参数"""
        return self.noise_params.get(self.noise_level, self.noise_params['medium'])
    
    def _random_param(self, param_range: Tuple[float, float]) -> float:
        """在参数范围内随机取值"""
        return random.uniform(param_range[0], param_range[1])
    
    def add_gaussian_noise(self, 
                          image: np.ndarray, 
                          sigma_range: Tuple[float, float] = None) -> np.ndarray:
        """
        添加高斯噪声
        
        Args:
            image: 输入图像 (H, W, C)
            sigma_range: 噪声标准差范围
            
        Returns:
            添加噪声后的图像
        """
        if sigma_range is None:
            params = self._get_params()
            sigma_range = params['gaussian_sigma']
        
        sigma = self._random_param(sigma_range)
        noise = np.random.normal(0, sigma, image.shape)
        noisy_image = image.astype(np.float32) + noise
        noisy_image = np.clip(noisy_image, 0, 255)
        return noisy_image.astype(np.uint8)
    
    def add_salt_pepper_noise(self, 
                             image: np.ndarray, 
                             prob_range: Tuple[float, float] = None) -> np.ndarray:
        """
        添加椒盐噪声
        
        Args:
            image: 输入图像
            prob_range: 椒盐噪声概率范围 (salt_prob, pepper_prob)
            
        Returns:
            添加噪声后的图像
        """
        if prob_range is None:
            params = self._get_params()
            prob_range = params['salt_pepper_prob']
        
        salt_prob = self._random_param(prob_range)
        pepper_prob = self._random_param(prob_range)
        
        noisy_image = image.copy()
        total_pixels = image.size // 3
        
        # 添加盐噪声（白点）
        num_salt = int(total_pixels * salt_prob)
        salt_coords = [np.random.randint(0, i, num_salt) for i in image.shape[:2]]
        noisy_image[salt_coords[0], salt_coords[1], :] = 255
        
        # 添加椒噪声（黑点）
        num_pepper = int(total_pixels * pepper_prob)
        pepper_coords = [np.random.randint(0, i, num_pepper) for i in image.shape[:2]]
        noisy_image[pepper_coords[0], pepper_coords[1], :] = 0
        
        return noisy_image
    
    def add_speckle_noise(self, 
                         image: np.ndarray, 
                         sigma_range: Tuple[float, float] = None) -> np.ndarray:
        """
        添加斑点噪声（乘性噪声）
        
        Args:
            image: 输入图像
            sigma_range: 噪声标准差范围
            
        Returns:
            添加噪声后的图像
        """
        if sigma_range is None:
            params = self._get_params()
            sigma_range = params['speckle_sigma']
        
        sigma = self._random_param(sigma_range)
        speckle = np.random.randn(*image.shape) * sigma
        noisy_image = image.astype(np.float32) + image.astype(np.float32) * speckle
        noisy_image = np.clip(noisy_image, 0, 255)
        return noisy_image.astype(np.uint8)
    
    def add_scan_lines(self, image: np.ndarray) -> np.ndarray:
        """
        添加扫描线效果
        
        Args:
            image: 输入图像
            
        Returns:
            添加扫描线后的图像
        """
        h, w = image.shape[:2]
        result = image.copy()
        
        # 水平扫描线
        if random.random() < 0.4:
            num_lines = random.randint(1, 3)
            for _ in range(num_lines):
                line_pos = random.randint(0, h-1)
                line_width = random.randint(1, 2)
                line_intensity = random.randint(220, 255)
                result[line_pos:line_pos+line_width, :, :] = line_intensity
        
        # 垂直扫描线（较少见）
        if random.random() < 0.1:
            line_pos = random.randint(0, w-1)
            line_width = random.randint(1, 2)
            line_intensity = random.randint(220, 255)
            result[:, line_pos:line_pos+line_width, :] = line_intensity
        
        return result
    
    def add_shadow_edges(self, image: np.ndarray) -> np.ndarray:
        """
        添加边缘阴影（模拟扫描仪边缘）
        
        Args:
            image: 输入图像
            
        Returns:
            添加阴影后的图像
        """
        h, w = image.shape[:2]
        result = image.copy().astype(np.float32)
        
        # 创建阴影遮罩
        shadow_mask = np.ones((h, w), dtype=np.float32)
        
        # 随机选择阴影位置
        edges = ['top', 'bottom', 'left', 'right']
        selected_edges = random.sample(edges, k=random.randint(1, 2))
        
        for edge in selected_edges:
            shadow_width = random.randint(int(w*0.02), int(w*0.08))
            shadow_intensity = random.uniform(0.7, 0.95)
            
            if edge == 'top':
                for i in range(shadow_width):
                    weight = 1 - (shadow_intensity * (shadow_width - i) / shadow_width)
                    shadow_mask[i, :] *= weight
            elif edge == 'bottom':
                for i in range(shadow_width):
                    weight = 1 - (shadow_intensity * (shadow_width - i) / shadow_width)
                    shadow_mask[h-1-i, :] *= weight
            elif edge == 'left':
                for j in range(shadow_width):
                    weight = 1 - (shadow_intensity * (shadow_width - j) / shadow_width)
                    shadow_mask[:, j] *= weight
            elif edge == 'right':
                for j in range(shadow_width):
                    weight = 1 - (shadow_intensity * (shadow_width - j) / shadow_width)
                    shadow_mask[:, w-1-j] *= weight
        
        # 应用阴影
        for c in range(3):
            result[:, :, c] *= shadow_mask
        
        result = np.clip(result, 0, 255)
        return result.astype(np.uint8)
    
    def add_paper_texture(self, image: np.ndarray) -> np.ndarray:
        """
        添加纸张纹理效果
        
        Args:
            image: 输入图像
            
        Returns:
            添加纹理后的图像
        """
        h, w = image.shape[:2]
        result = image.copy().astype(np.float32)
        
        # 创建纸张纹理
        texture = np.ones((h, w), dtype=np.float32)
        
        # 添加细微的纹理噪声
        texture_noise = np.random.randn(h, w) * 0.05
        texture += texture_noise
        
        # 添加纸张纤维效果
        if random.random() < 0.3:
            fiber_strength = random.uniform(0.01, 0.03)
            for _ in range(random.randint(2, 5)):
                fiber_length = random.randint(int(h*0.1), int(h*0.3))
                fiber_width = random.randint(1, 3)
                start_x = random.randint(0, w-1)
                start_y = random.randint(0, h-1)
                angle = random.uniform(0, math.pi)
                
                for i in range(fiber_length):
                    x = int(start_x + i * math.cos(angle))
                    y = int(start_y + i * math.sin(angle))
                    
                    if 0 <= x < w and 0 <= y < h:
                        for dx in range(-fiber_width, fiber_width+1):
                            for dy in range(-fiber_width, fiber_width+1):
                                nx, ny = x + dx, y + dy
                                if 0 <= nx < w and 0 <= ny < h:
                                    dist = math.sqrt(dx*dx + dy*dy)
                                    if dist <= fiber_width:
                                        texture[ny, nx] += fiber_strength * (1 - dist/fiber_width)
        
        # 应用纹理
        for c in range(3):
            result[:, :, c] *= texture
        
        result = np.clip(result, 0, 255)
        return result.astype(np.uint8)
    
    def add_ink_bleed(self, image: np.ndarray) -> np.ndarray:
        """
        添加墨水扩散效果
        
        Args:
            image: 输入图像
            
        Returns:
            添加墨水扩散后的图像
        """
        result = image.copy()
        
        # 只在深色区域添加墨水扩散
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
        
        # 膨胀操作模拟墨水扩散
        kernel_size = random.randint(1, 2)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        dilated = cv2.dilate(binary, kernel, iterations=1)
        
        # 将扩散区域应用到原图
        mask = dilated > 0
        for c in range(3):
            channel = result[:, :, c]
            channel[mask] = np.clip(channel[mask].astype(np.int32) - random.randint(5, 15), 0, 255)
            result[:, :, c] = channel
        
        return result
    
    def add_stain_spots(self, image: np.ndarray) -> np.ndarray:
        """
        添加污渍斑点
        
        Args:
            image: 输入图像
            
        Returns:
            添加污渍后的图像
        """
        result = image.copy()
        h, w = image.shape[:2]
        
        # 随机添加污渍
        num_stains = random.randint(0, 3)
        for _ in range(num_stains):
            stain_type = random.choice(['dark', 'light', 'color'])
            radius = random.randint(int(min(h, w)*0.02), int(min(h, w)*0.08))
            center_x = random.randint(radius, w - radius - 1)
            center_y = random.randint(radius, h - radius - 1)
            
            # 创建圆形遮罩
            y, x = np.ogrid[:h, :w]
            mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
            
            if stain_type == 'dark':
                # 深色污渍
                stain_value = random.randint(50, 100)
                for c in range(3):
                    result[:, :, c][mask] = np.minimum(result[:, :, c][mask], stain_value)
            elif stain_type == 'light':
                # 浅色污渍
                stain_value = random.randint(180, 230)
                for c in range(3):
                    result[:, :, c][mask] = np.maximum(result[:, :, c][mask], stain_value)
            else:
                # 彩色污渍
                stain_color = [random.randint(150, 250) for _ in range(3)]
                for c in range(3):
                    result[:, :, c][mask] = stain_color[c]
        
        return result
    
    def apply_blur(self, image: np.ndarray) -> np.ndarray:
        """
        应用模糊效果（模拟失焦）
        
        Args:
            image: 输入图像
            
        Returns:
            模糊后的图像
        """
        params = self._get_params()
        kernel_range = params['blur_kernel']
        kernel_size = random.randint(kernel_range[0], kernel_range[1])
        
        # 确保核大小为奇数
        kernel_size = kernel_size * 2 + 1
        
        blur_type = random.choice(['gaussian', 'median', 'motion'])
        
        if blur_type == 'gaussian':
            return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        elif blur_type == 'median':
            return cv2.medianBlur(image, kernel_size)
        else:  # motion blur
            # 创建运动模糊核
            kernel_motion_blur = np.zeros((kernel_size, kernel_size))
            kernel_motion_blur[int((kernel_size-1)/2), :] = np.ones(kernel_size)
            kernel_motion_blur = kernel_motion_blur / kernel_size
            
            # 随机旋转核
            angle = random.uniform(0, 180)
            M = cv2.getRotationMatrix2D((kernel_size/2, kernel_size/2), angle, 1)
            kernel_motion_blur = cv2.warpAffine(kernel_motion_blur, M, (kernel_size, kernel_size))
            
            return cv2.filter2D(image, -1, kernel_motion_blur)
    
    def adjust_brightness_contrast(self, image: np.ndarray) -> np.ndarray:
        """
        调整亮度和对比度
        
        Args:
            image: 输入图像
            
        Returns:
            调整后的图像
        """
        params = self._get_params()
        brightness_range = params['brightness_range']
        contrast_range = params['contrast_range']
        
        brightness = random.randint(brightness_range[0], brightness_range[1])
        contrast = self._random_param(contrast_range)
        
        # 调整对比度
        result = image.astype(np.float32)
        result = contrast * (result - 127.5) + 127.5
        
        # 调整亮度
        result += brightness
        
        result = np.clip(result, 0, 255)
        return result.astype(np.uint8)
    
    def apply_perspective_transform(self, image: np.ndarray) -> np.ndarray:
        """
        应用透视变换（模拟纸张不平）
        
        Args:
            image: 输入图像
            
        Returns:
            变换后的图像
        """
        h, w = image.shape[:2]
        
        # 原始四角点
        pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        
        # 随机扰动四角点
        max_offset = int(min(h, w) * 0.02)
        pts2 = np.float32([
            [random.randint(-max_offset, max_offset), 
             random.randint(-max_offset, max_offset)],
            [w + random.randint(-max_offset, max_offset), 
             random.randint(-max_offset, max_offset)],
            [random.randint(-max_offset, max_offset), 
             h + random.randint(-max_offset, max_offset)],
            [w + random.randint(-max_offset, max_offset), 
             h + random.randint(-max_offset, max_offset)]
        ])
        
        # 计算透视变换矩阵
        M = cv2.getPerspectiveTransform(pts1, pts2)
        
        # 应用变换
        transformed = cv2.warpPerspective(image, M, (w, h), 
                                         borderMode=cv2.BORDER_CONSTANT,
                                         borderValue=(255, 255, 255))
        
        return transformed
    
    def apply_rotation(self, image: np.ndarray) -> np.ndarray:
        """
        应用旋转（模拟扫描不正）
        
        Args:
            image: 输入图像
            
        Returns:
            旋转后的图像
        """
        h, w = image.shape[:2]
        
        # 小角度旋转
        angle = random.uniform(-2, 2)
        
        # 计算旋转矩阵
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # 计算新边界
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        
        # 调整旋转矩阵
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]
        
        # 应用旋转
        rotated = cv2.warpAffine(image, M, (new_w, new_h),
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=(255, 255, 255))
        
        # 裁剪回原始大小
        start_x = (new_w - w) // 2
        start_y = (new_h - h) // 2
        cropped = rotated[start_y:start_y+h, start_x:start_x+w]
        
        return cropped
    
    def __call__(self, image: Union[np.ndarray, Image.Image]) -> Image.Image:
        """
        应用所有增强
        
        Args:
            image: 输入图像 (PIL Image 或 numpy array)
            
        Returns:
            增强后的PIL Image
        """
        # 决定是否应用增强
        if random.random() > self.apply_prob:
            if isinstance(image, Image.Image):
                return image
            else:
                return Image.fromarray(image)
        
        # 转换为numpy数组
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # 保存原始图像用于某些操作
        original = image.copy()
        
        # 根据扫描类型选择增强序列
        if self.scan_type == 'document':
            # 文档扫描：轻微噪声和阴影
            sequence = [
                (0.8, self.adjust_brightness_contrast),
                (0.6, self.add_shadow_edges),
                (0.5, self.add_gaussian_noise),
                (0.3, self.apply_blur),
                (0.2, self.apply_rotation),
            ]
        elif self.scan_type == 'book':
            # 书籍扫描：阴影、纹理和弯曲
            sequence = [
                (0.9, self.add_shadow_edges),
                (0.7, self.add_paper_texture),
                (0.5, self.adjust_brightness_contrast),
                (0.4, self.apply_perspective_transform),
                (0.3, self.add_speckle_noise),
                (0.2, self.add_ink_bleed),
            ]
        elif self.scan_type == 'old_paper':
            # 旧纸张：污渍、纹理和噪声
            sequence = [
                (0.8, self.add_paper_texture),
                (0.6, self.add_stain_spots),
                (0.5, self.add_salt_pepper_noise),
                (0.4, self.add_ink_bleed),
                (0.3, self.adjust_brightness_contrast),
                (0.2, self.add_scan_lines),
            ]
        else:  # mixed
            # 混合增强：随机选择
            all_augmentations = [
                (0.5, self.add_gaussian_noise),
                (0.3, self.add_salt_pepper_noise),
                (0.3, self.add_speckle_noise),
                (0.4, self.add_scan_lines),
                (0.4, self.add_shadow_edges),
                (0.3, self.add_paper_texture),
                (0.2, self.add_ink_bleed),
                (0.2, self.add_stain_spots),
                (0.4, self.apply_blur),
                (0.5, self.adjust_brightness_contrast),
                (0.2, self.apply_perspective_transform),
                (0.2, self.apply_rotation),
            ]
            # 随机选择3-6种增强
            num_augs = random.randint(3, 6)
            sequence = random.sample(all_augmentations, num_augs)
        
        # 按顺序应用增强
        for prob, aug_func in sequence:
            if random.random() < prob:
                try:
                    image = aug_func(image)
                except Exception as e:
                    # 如果增强失败，使用原始图像继续
                    print(f"Warning: Augmentation failed: {e}")
                    image = original.copy()
        
        # 确保图像在有效范围内
        image = np.clip(image, 0, 255).astype(np.uint8)
        
        # 转换为PIL Image
        return Image.fromarray(image)


def process_single_image(args):
    """处理单张图像的辅助函数（用于多进程）"""
    input_path, output_path, augmentor = args
    
    try:
        # 加载图像
        image = Image.open(input_path).convert('RGB')
        
        # 应用增强
        augmented_image = augmentor(image)
        
        # 保存图像
        augmented_image.save(output_path)
        return True, input_path, None
    except Exception as e:
        return False, input_path, str(e)


def batch_process_images(input_dir: str, 
                        output_dir: str,
                        apply_prob: float = 0.7,
                        noise_level: str = 'medium',
                        scan_type: str = 'mixed',
                        num_workers: int = 4,
                        num_augmentations: int = 1,
                        preserve_structure: bool = True):
    """
    批量处理文件夹中的所有图片
    
    Args:
        input_dir: 输入图片文件夹路径
        output_dir: 输出图片文件夹路径
        apply_prob: 应用增强的概率
        noise_level: 噪声级别
        scan_type: 扫描类型
        num_workers: 多进程工作数
        num_augmentations: 每张图片生成多少个增强版本
        preserve_structure: 是否保持原始文件夹结构
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化增强器
    augmentor = FormulaScanAugmentation(
        apply_prob=apply_prob,
        noise_level=noise_level,
        scan_type=scan_type
    )
    
    # 获取所有图片文件
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif', '*.gif']
    image_paths = []
    
    for ext in image_extensions:
        pattern = os.path.join(input_dir, '**', ext) if preserve_structure else os.path.join(input_dir, ext)
        image_paths.extend(glob.glob(pattern, recursive=preserve_structure))
    
    # 去除重复（可能由于模式匹配）
    image_paths = list(set(image_paths))
    
    if not image_paths:
        print(f"警告: 在 {input_dir} 中没有找到图片文件")
        return
    
    print(f"找到 {len(image_paths)} 张图片")
    
    # 准备处理任务
    tasks = []
    
    for input_path in image_paths:
        # 获取相对路径（用于保持目录结构）
        if preserve_structure:
            rel_path = os.path.relpath(input_path, input_dir)
        else:
            rel_path = os.path.basename(input_path)
        
        # 为每张图片生成指定数量的增强版本
        for aug_idx in range(num_augmentations):
            # 生成输出文件名
            if num_augmentations > 1:
                # 如果有多个增强版本，添加编号
                base_name, ext = os.path.splitext(rel_path)
                output_filename = f"{base_name}_aug{aug_idx:03d}{ext}"
            else:
                output_filename = rel_path
            
            # 创建输出路径
            output_path = os.path.join(output_dir, output_filename)
            
            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            tasks.append((input_path, output_path, augmentor))
    
    print(f"将生成 {len(tasks)} 个增强图像")
    
    # 使用多进程处理
    if num_workers > 1:
        print(f"使用 {num_workers} 个进程进行批量处理...")
        
        # 创建进程池
        with multiprocessing.Pool(processes=num_workers) as pool:
            results = list(tqdm(pool.imap(process_single_image, tasks), 
                              total=len(tasks), 
                              desc="处理图片"))
    else:
        print("使用单进程进行批量处理...")
        results = []
        for task in tqdm(tasks, desc="处理图片"):
            results.append(process_single_image(task))
    
    # 统计结果
    success_count = 0
    error_count = 0
    errors = []
    
    for success, input_path, error_msg in results:
        if success:
            success_count += 1
        else:
            error_count += 1
            errors.append((input_path, error_msg))
    
    # 输出统计信息
    print("\n" + "="*50)
    print("批量处理完成!")
    print(f"成功处理: {success_count} 张图片")
    print(f"处理失败: {error_count} 张图片")
    
    if errors:
        print("\n失败详情:")
        for input_path, error_msg in errors[:10]:  # 只显示前10个错误
            print(f"  {os.path.basename(input_path)}: {error_msg}")
        if len(errors) > 10:
            print(f"  ... 还有 {len(errors) - 10} 个错误未显示")


def visualize_augmentation_samples(input_dir: str, 
                                  output_dir: str,
                                  num_samples: int = 5,
                                  apply_prob: float = 1.0,
                                  noise_level: str = 'medium',
                                  scan_type: str = 'mixed'):
    """
    可视化增强效果示例
    
    Args:
        input_dir: 输入图片文件夹
        output_dir: 输出可视化图片文件夹
        num_samples: 显示多少张示例
        apply_prob: 应用增强的概率
        noise_level: 噪声级别
        scan_type: 扫描类型
    """
    import matplotlib.pyplot as plt
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化增强器
    augmentor = FormulaScanAugmentation(
        apply_prob=apply_prob,
        noise_level=noise_level,
        scan_type=scan_type
    )
    
    # 获取图片文件
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_paths = []
    
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(input_dir, ext)))
    
    if not image_paths:
        print(f"警告: 在 {input_dir} 中没有找到图片文件")
        return
    
    # 随机选择示例图片
    selected_paths = random.sample(image_paths, min(num_samples, len(image_paths)))
    
    for idx, img_path in enumerate(selected_paths):
        try:
            # 加载原始图像
            original = Image.open(img_path).convert('RGB')
            
            # 创建多个增强版本
            augmented_images = []
            for i in range(4):  # 生成4个不同的增强版本
                image = Image.open(img_path).convert('RGB')
                augmented = augmentor(image)
                augmented_images.append(augmented)
            
            # 创建可视化图
            fig, axes = plt.subplots(1, 5, figsize=(20, 4))
            
            # 原始图像
            axes[0].imshow(original)
            axes[0].set_title('Original')
            axes[0].axis('off')
            
            # 增强图像
            for i, img in enumerate(augmented_images, 1):
                axes[i].imshow(img)
                axes[i].set_title(f'Augmented {i}')
                axes[i].axis('off')
            
            # 设置标题
            img_name = os.path.basename(img_path)
            plt.suptitle(f'Sample {idx+1}: {img_name}\n'
                        f'Noise Level: {noise_level}, Scan Type: {scan_type}', 
                        fontsize=12)
            plt.tight_layout()
            
            # 保存可视化图
            output_path = os.path.join(output_dir, f"sample_{idx+1:03d}.png")
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            print(f"已保存可视化示例: {output_path}")
            
        except Exception as e:
            print(f"处理 {img_path} 时出错: {e}")


def main():
    """主函数：命令行接口"""
    parser = argparse.ArgumentParser(description='公式图像扫描效果数据增强工具')
    parser.add_argument('--input', '-i', type=str, required=True,
                       help='输入图片文件夹路径')
    parser.add_argument('--output', '-o', type=str, required=True,
                       help='输出图片文件夹路径')
    parser.add_argument('--apply-prob', type=float, default=0.7,
                       help='应用增强的概率 (0-1)，默认: 0.7')
    parser.add_argument('--noise-level', type=str, default='medium',
                       choices=['light', 'medium', 'heavy'],
                       help='噪声级别，默认: medium')
    parser.add_argument('--scan-type', type=str, default='mixed',
                       choices=['document', 'book', 'old_paper', 'mixed'],
                       help='扫描类型，默认: mixed')
    parser.add_argument('--num-augmentations', '-n', type=int, default=1,
                       help='每张图片生成多少个增强版本，默认: 1')
    parser.add_argument('--workers', '-w', type=int, default=4,
                       help='多进程工作数，默认: 4')
    parser.add_argument('--flat', action='store_true',
                       help='不保持原始文件夹结构，所有输出到同一目录')
    parser.add_argument('--visualize', action='store_true',
                       help='生成增强效果可视化示例')
    parser.add_argument('--visualize-only', action='store_true',
                       help='只生成可视化示例，不进行批量处理')
    parser.add_argument('--visualize-samples', type=int, default=5,
                       help='可视化示例数量，默认: 5')
    
    args = parser.parse_args()
    
    # 检查输入目录是否存在
    if not os.path.exists(args.input):
        print(f"错误: 输入目录不存在: {args.input}")
        return
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    
    # 生成可视化示例
    if args.visualize or args.visualize_only:
        print("生成增强效果可视化示例...")
        visualize_dir = os.path.join(args.output, "visualization")
        visualize_augmentation_samples(
            input_dir=args.input,
            output_dir=visualize_dir,
            num_samples=args.visualize_samples,
            apply_prob=1.0,  # 可视化时总是应用增强
            noise_level=args.noise_level,
            scan_type=args.scan_type
        )
        print(f"可视化示例已保存到: {visualize_dir}")
    
    # 如果不只是可视化，进行批量处理
    if not args.visualize_only:
        print("\n开始批量处理图片...")
        batch_process_images(
            input_dir=args.input,
            output_dir=args.output,
            apply_prob=args.apply_prob,
            noise_level=args.noise_level,
            scan_type=args.scan_type,
            num_workers=args.workers,
            num_augmentations=args.num_augmentations,
            preserve_structure=not args.flat
        )
    
    print("\n处理完成！")


if __name__ == "__main__":
    main()

'''
    # 基本用法：处理文件夹中的所有图片
    python formula_augmentation.py --input ./input_images --output ./augmented_images
    
    # 指定增强参数
    python formula_augmentation.py --input ./input_images --output ./augmented_images \
        --noise-level heavy \
        --scan-type old_paper \
        --apply-prob 0.8
    
    # 每张图片生成3个增强版本
    python formula_augmentation.py --input ./input_images --output ./augmented_images \
        --num-augmentations 3 \
        --workers 8  # 使用8个进程
    
    # 不保持文件夹结构（所有输出到同一目录）
    python formula_augmentation.py --input ./input_images --output ./augmented_images --flat
    
    # 生成可视化示例
    python formula_augmentation.py --input ./input_images --output ./augmented_images --visualize
    
    # 只生成可视化示例，不进行批量处理
    python formula_augmentation.py --input ./input_images --output ./augmented_images --visualize-only
'''