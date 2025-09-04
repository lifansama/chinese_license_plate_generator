#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OCR标签生成脚本
从车牌文件名生成OCR训练所需的标签文件
支持多线程处理和数据集分割

默认处理 plate/single 和 plate/double 目录下的车牌图片。
可通过 --subdirs 参数指定其他目录，或使用 --all-dirs 处理整个源目录。
"""

import os
import argparse
import random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from collections import defaultdict
import json


# 全局线程锁
label_lock = threading.Lock()


def extract_plate_text_from_filename(filename):
    """
    从文件名中提取车牌文字内容
    
    Args:
        filename: 文件名，格式如 京A3340警_white_single_1_police-1-(352).jpg
        
    Returns:
        str: 车牌文字内容，去除下划线
    """
    # 移除文件扩展名
    name_without_ext = os.path.splitext(filename)[0]
    
    try:
        # 按下划线分割文件名
        parts = name_without_ext.split('_')
        
        if len(parts) < 3:
            print(f"警告: 文件名格式不正确: {filename}")
            return None
        
        # 检查是否为双层车牌且车牌号码部分包含下划线
        if parts[2] in ['single', 'double', 'unknownnull']:
            # 标准格式：车牌号码_颜色_层数_...
            plate_number = parts[0]
        elif len(parts) >= 4 and parts[3] in ['single', 'double', 'unknownnull']:
            # 双层车牌带下划线格式：京A_3340警_颜色_层数_...
            # 需要将前两部分合并为车牌号码
            plate_number = parts[0] + '_' + parts[1]
        else:
            print(f"警告: 无法识别文件名格式: {filename}")
            return None
        
        # 去除车牌号码中的下划线（双层车牌可能有下划线）
        plate_text = plate_number.replace('_', '')
        
        return plate_text
        
    except Exception as e:
        print(f"解析文件名失败: {filename}, 错误: {e}")
        return None


def is_image_file(filename):
    """检查是否为图片文件"""
    return filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))


def collect_plate_files(directory, max_workers=4, target_subdirs=None):
    """
    收集车牌文件并按车牌号分组
    
    Args:
        directory: 目录路径
        max_workers: 线程数
        target_subdirs: 目标子目录列表，如果为None则处理整个目录
        
    Returns:
        dict: {plate_text: [relative_paths]}
    """
    # 设置默认目标子目录
    if target_subdirs is None:
        # None 表示处理整个目录，不设置默认值
        pass
    elif len(target_subdirs) == 0:
        # 空列表使用默认子目录
        target_subdirs = ['plate/single', 'plate/double']
    
    print(f"正在扫描目录: {directory}")
    if target_subdirs is None:
        print("目标: 整个目录")
    else:
        print(f"目标子目录: {target_subdirs}")
    
    # 收集所有图片文件
    all_image_files = []
    
    def scan_directory(scan_dir):
        """递归扫描目录"""
        for root, dirs, files in os.walk(scan_dir):
            for filename in files:
                if is_image_file(filename):
                    full_path = os.path.join(root, filename)
                    # 计算相对于给定目录的相对路径
                    rel_path = os.path.relpath(full_path, directory)
                    all_image_files.append((rel_path, filename))
    
    # 扫描指定的子目录或整个目录
    if target_subdirs is None:
        # 扫描整个目录
        print("扫描整个目录")
        scan_directory(directory)
    else:
        # 扫描指定的子目录
        for subdir in target_subdirs:
            full_subdir_path = os.path.join(directory, subdir)
            if os.path.exists(full_subdir_path):
                print(f"扫描子目录: {subdir}")
                scan_directory(full_subdir_path)
            else:
                print(f"警告: 子目录不存在: {subdir}")
    
    print(f"找到 {len(all_image_files)} 个图片文件")
    
    # 多线程处理文件，按车牌号分组
    plate_groups = defaultdict(list)
    
    def process_file_batch(file_batch):
        """处理一批文件"""
        local_groups = defaultdict(list)
        
        for rel_path, filename in file_batch:
            plate_text = extract_plate_text_from_filename(filename)
            if plate_text:
                local_groups[plate_text].append(rel_path)
        
        return local_groups
    
    # 将文件分批处理
    batch_size = max(1, len(all_image_files) // max_workers)
    batches = [all_image_files[i:i + batch_size] 
               for i in range(0, len(all_image_files), batch_size)]
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_batch = {executor.submit(process_file_batch, batch): batch 
                          for batch in batches}
        
        for future in as_completed(future_to_batch):
            local_groups = future.result()
            
            # 合并结果
            with label_lock:
                for plate_text, paths in local_groups.items():
                    plate_groups[plate_text].extend(paths)
    
    print(f"识别出 {len(plate_groups)} 个不同的车牌")
    return dict(plate_groups)


def generate_label_line(paths, plate_text):
    """
    生成单行标签
    
    Args:
        paths: 文件路径列表
        plate_text: 车牌文字
        
    Returns:
        str: 标签行
    """
    if len(paths) == 1:
        return f"{paths[0]}\t{plate_text}"
    else:
        # 多个文件使用JSON数组格式
        paths_json = json.dumps(paths, ensure_ascii=False)
        return f"{paths_json}\t{plate_text}"


def write_labels_file(plate_groups, output_file):
    """
    写入标签文件
    
    Args:
        plate_groups: {plate_text: [paths]}
        output_file: 输出文件路径
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for plate_text, paths in sorted(plate_groups.items()):
            # 对同一车牌的多个文件路径进行排序
            paths = sorted(paths)
            line = generate_label_line(paths, plate_text)
            f.write(line + '\n')
    
    print(f"标签文件已保存到: {output_file}")


def split_dataset(all_file, train_ratio=0.9, val_ratio=0.1, random_seed=42):
    """
    分割数据集
    
    Args:
        all_file: 包含所有标签的文件路径
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        random_seed: 随机种子
    """
    if abs(train_ratio + val_ratio - 1.0) > 1e-6:
        raise ValueError("训练集和验证集比例之和必须等于1.0")
    
    # 读取所有标签
    with open(all_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 随机打乱
    random.seed(random_seed)
    random.shuffle(lines)
    
    # 计算分割点
    total_lines = len(lines)
    train_split = int(total_lines * train_ratio)
    
    train_lines = lines[:train_split]
    val_lines = lines[train_split:]
    
    # 写入训练集
    train_file = os.path.join(os.path.dirname(all_file), 'train.txt')
    with open(train_file, 'w', encoding='utf-8') as f:
        f.writelines(train_lines)
    
    # 写入验证集
    val_file = os.path.join(os.path.dirname(all_file), 'val.txt')
    with open(val_file, 'w', encoding='utf-8') as f:
        f.writelines(val_lines)
    
    print(f"数据集分割完成:")
    print(f"  训练集: {len(train_lines)} 条 -> {train_file}")
    print(f"  验证集: {len(val_lines)} 条 -> {val_file}")


def main():
    parser = argparse.ArgumentParser(description='OCR标签生成脚本')
    parser.add_argument('--source_dir', default='./', help='源目录路径（包含车牌图片的目录）')
    parser.add_argument('--threads', type=int, default=16, help='线程数，默认为16')
    parser.add_argument('--split', action='store_true', help='是否分割数据集')
    parser.add_argument('--val-ratio', type=float, default=0.1, help='验证集比例，默认0.1')
    parser.add_argument('--train-ratio', type=float, default=0.9, help='训练集比例，默认0.9')
    parser.add_argument('--seed', type=int, default=42, help='随机种子，默认42')
    parser.add_argument('--dry-run', action='store_true', help='预览模式，只显示处理结果不写入文件')
    parser.add_argument('--subdirs', nargs='*', default=['plate/single', 'plate/double_preprocessed'], 
                       help='要处理的子目录列表，默认为 plate/single 和 plate/double_preprocessed')
    parser.add_argument('--all-dirs', action='store_true', help='处理整个源目录（忽略--subdirs参数）')
    
    args = parser.parse_args()
    
    source_dir = os.path.abspath(args.source_dir)
    
    # 验证源目录
    if not os.path.exists(source_dir):
        print(f"错误: 源目录不存在: {source_dir}")
        return
    
    # 验证分割比例
    if args.split and abs(args.train_ratio + args.val_ratio - 1.0) > 1e-6:
        print(f"错误: 训练集比例({args.train_ratio})和验证集比例({args.val_ratio})之和必须等于1.0")
        return
    
    print("=== OCR标签生成 ===")
    print(f"源目录: {source_dir}")
    print(f"线程数: {args.threads}")
    
    # 确定要处理的目录
    if args.all_dirs:
        target_subdirs = None
    else:
        target_subdirs = args.subdirs if args.subdirs else ['plate/single', 'plate/double']
    
    print(f"处理模式: {'全目录' if args.all_dirs else '指定子目录'}")
    if not args.all_dirs:
        print(f"子目录列表: {target_subdirs}")
    
    # 收集车牌文件
    plate_groups = collect_plate_files(source_dir, args.threads, target_subdirs)
    
    if not plate_groups:
        print("没有找到有效的车牌文件")
        return
    
    # 统计信息
    total_plates = len(plate_groups)
    total_images = sum(len(paths) for paths in plate_groups.values())
    multi_image_plates = sum(1 for paths in plate_groups.values() if len(paths) > 1)
    
    print(f"\n统计信息:")
    print(f"  车牌数量: {total_plates}")
    print(f"  图片数量: {total_images}")
    print(f"  多图片车牌: {multi_image_plates}")
    
    if args.dry_run:
        print("\n=== 预览模式 ===")
        print("标签示例（前10条）:")
        count = 0
        for plate_text, paths in sorted(plate_groups.items()):
            if count >= 10:
                print("...")
                break
            paths = sorted(paths)
            line = generate_label_line(paths, plate_text)
            print(line)
            count += 1
        return
    
    # 生成输出文件路径
    output_dir = os.path.join(source_dir, 'rec')
    all_file = os.path.join(output_dir, 'all.txt')
    
    # 写入标签文件
    write_labels_file(plate_groups, all_file)
    
    # 分割数据集
    if args.split:
        print("\n=== 数据集分割 ===")
        split_dataset(all_file, args.train_ratio, args.val_ratio, args.seed)
    
    print("\n处理完成!")


if __name__ == "__main__":
    main()
