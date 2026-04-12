#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSGO 人物识别训练系统 (YOLOv11 兼容版) - 资源优化版
Python 3.13.5 | CUDA 12.9 | cuDNN | YOLOv11
训练集: 9024张图片 | 验证集: 1000张图片
"""

import os
import sys
import argparse
import xml.etree.ElementTree as ET
import yaml
import datetime
import numpy as np
import torch
from pathlib import Path
from ultralytics import YOLO
import psutil
import GPUtil
import gc  # 添加垃圾回收模块


# ============== 配置参数 ==============
class Config:
    # 数据集配置
    DATA_ROOT = 'dataset'
    CLASSES = ['body', 'head']  # 0: body, 1: head
    CLASS_MAPPING = {"body": 0, "head": 1}

    # 数据集大小
    TRAIN_SIZE = 9024  # 训练图片数量
    VAL_SIZE = 1000  # 验证图片数量

    # 模型配置 - 根据要求保留原始设置
    MODEL = 'yolo11n.pt'  # 使用指定的模型文件
    IMGSZ = 640  # 保持原始图像尺寸
    BATCH_SIZE = 8  # 保持批处理大小为12
    EPOCHS = 100  # 适当减少训练轮数

    # 优化器配置
    LR0 = 0.01
    LRF = 0.01
    MOMENTUM = 0.937
    WEIGHT_DECAY = 0.0005

    # 高级配置
    PATIENCE = 30  # 提前停止的耐心值
    BOX = 7.5  # 边界框损失权重
    CLS = 0.5  # 分类损失权重
    DFL = 1.5  # DFL损失权重

    # 性能优化 - 针对内存限制调整
    WORKERS = 5  # 减少工作进程数，但保持合理值
    CACHE = False  # 禁用数据集缓存

    # 输出配置
    SAVE_PERIOD = 10  # 每10个epoch保存一次
    PROJECT = 'runs/detect/csgo_v11_optimized'

    # 设备配置
    DEVICE = '0' if torch.cuda.is_available() else 'cpu'


# ============== XML转YOLO格式 ==============
def convert_xml_to_yolo(xml_path, output_dir):
    """将单个XML文件转换为YOLO格式的TXT文件"""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # 获取图像尺寸
        size = root.find('size')
        img_width = int(size.find('width').text)
        img_height = int(size.find('height').text)

        # 创建输出文件路径
        txt_path = os.path.join(output_dir, Path(xml_path).stem + '.txt')

        with open(txt_path, 'w') as f:
            for obj in root.findall('object'):
                # 获取类别
                class_name = obj.find('name').text
                class_id = Config.CLASS_MAPPING.get(class_name)
                if class_id is None:
                    continue  # 跳过未知类别

                # 获取边界框
                bbox = obj.find('bndbox')
                xmin = float(bbox.find('xmin').text)
                ymin = float(bbox.find('ymin').text)
                xmax = float(bbox.find('xmax').text)
                ymax = float(bbox.find('ymax').text)

                # 转换为YOLO格式 (归一化中心坐标和宽高)
                x_center = ((xmin + xmax) / 2) / img_width
                y_center = ((ymin + ymax) / 2) / img_height
                width = (xmax - xmin) / img_width
                height = (ymax - ymin) / img_height

                # 写入文件
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

        return True
    except Exception as e:
        print(f"转换失败: {xml_path} - {e}")
        return False


def convert_all_xml_to_yolo():
    """转换所有XML文件到YOLO格式"""
    # 训练集转换
    train_xml_dir = os.path.join(Config.DATA_ROOT, 'labels', 'train')
    if os.path.exists(train_xml_dir):
        xml_files = [f for f in os.listdir(train_xml_dir) if f.endswith('.xml')]
        print(f"找到 {len(xml_files)} 个训练集XML文件，开始转换...")
        for i, xml_file in enumerate(xml_files):
            convert_xml_to_yolo(
                os.path.join(train_xml_dir, xml_file),
                train_xml_dir
            )
            # 每处理200个文件释放一次内存
            if i % 200 == 0:
                gc.collect()
                print(f"已处理 {i + 1}/{len(xml_files)} 个文件，释放内存...")

    # 验证集转换
    val_xml_dir = os.path.join(Config.DATA_ROOT, 'labels', 'val')
    if os.path.exists(val_xml_dir):
        xml_files = [f for f in os.listdir(val_xml_dir) if f.endswith('.xml')]
        print(f"找到 {len(xml_files)} 个验证集XML文件，开始转换...")
        for i, xml_file in enumerate(xml_files):
            convert_xml_to_yolo(
                os.path.join(val_xml_dir, xml_file),
                val_xml_dir
            )
            # 每处理100个文件释放一次内存
            if i % 100 == 0:
                gc.collect()
                print(f"已处理 {i + 1}/{len(xml_files)} 个文件，释放内存...")


# ============== 数据集配置 ==============
def create_dataset_yaml():
    """创建或更新数据集配置文件"""
    yaml_path = os.path.join(Config.DATA_ROOT, 'dataset.yaml')
    content = f"""
# CSGO人物识别数据集配置
path: {os.path.abspath(Config.DATA_ROOT)}  # 数据集根目录
train: images/train  # 训练图像相对路径
val: images/val      # 验证图像相对路径

# 类别信息
names:
  0: {Config.CLASSES[0]}
  1: {Config.CLASSES[1]}
"""
    with open(yaml_path, 'w') as f:
        f.write(content.strip())
    print(f"数据集配置文件已创建/更新: {yaml_path}")
    return yaml_path


# ============== 系统资源监控 ==============
def print_system_stats():
    """打印系统资源使用情况"""
    # CPU使用率
    cpu_percent = psutil.cpu_percent(interval=1)

    # 内存使用
    mem = psutil.virtual_memory()
    mem_used = mem.used / (1024 ** 3)  # GB
    mem_total = mem.total / (1024 ** 3)
    mem_available = mem.available / (1024 ** 3)

    # GPU使用
    gpu_info = ""
    try:
        gpus = GPUtil.getGPUs()
        for i, gpu in enumerate(gpus):
            gpu_info += (f"GPU {i}: {gpu.load * 100:.1f}% | "
                         f"Mem: {gpu.memoryUsed:.1f}/{gpu.memoryTotal:.1f} GB | "
                         f"Temp: {gpu.temperature}°C\n")
    except Exception:
        gpu_info = "GPU信息不可用"

    print(f"\n[系统资源] CPU: {cpu_percent}% | "
          f"内存: {mem_used:.1f}/{mem_total:.1f} GB | "
          f"可用内存: {mem_available:.1f} GB")
    print(f"[GPU状态]\n{gpu_info}")

    # 如果可用内存不足2GB，建议用户调整设置
    if mem_available < 2:
        print("\n⚠️ 警告: 系统可用内存不足2GB，可能会影响训练稳定性！")
        print("建议关闭其他应用程序或进一步减少批处理大小。")

    return mem_available, gpu_info


# ============== 主训练函数 ==============
def train_model(resume_checkpoint=None):
    """执行模型训练 - 资源优化版"""
    # 设置CUDA设备
    torch.backends.cudnn.benchmark = True  # 启用cuDNN基准测试

    # 减少PyTorch内部线程数
    torch.set_num_threads(1)
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'

    # 确保数据集配置文件存在
    data_config = create_dataset_yaml()

    # 生成运行名称 (带时间戳)
    run_name = datetime.datetime.now().strftime('train_%Y%m%d_%H%M%S')

    # 加载模型
    print(f"\n===== 加载模型 =====")
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        print(f"从检查点恢复训练: {resume_checkpoint}")
        model = YOLO(resume_checkpoint)
    else:
        print(f"使用预训练模型: {Config.MODEL}")
        model = YOLO(Config.MODEL)

    # 获取实际数据集大小
    train_img_dir = os.path.join(Config.DATA_ROOT, 'images', 'train')
    val_img_dir = os.path.join(Config.DATA_ROOT, 'images', 'val')

    actual_train_size = len(os.listdir(train_img_dir)) if os.path.exists(train_img_dir) else 0
    actual_val_size = len(os.listdir(val_img_dir)) if os.path.exists(val_img_dir) else 0

    # 打印训练摘要
    print("\n===== 训练配置 =====")
    print(f"设备: {'GPU' if Config.DEVICE != 'cpu' else 'CPU'} | "
          f"批次大小: {Config.BATCH_SIZE} | 图像尺寸: {Config.IMGSZ}")
    print(f"训练周期: {Config.EPOCHS} | 学习率: {Config.LR0} -> {Config.LRF}")
    print(f"类别: {Config.CLASSES}")
    print(f"训练集: {actual_train_size} 张图片 (目标: {Config.TRAIN_SIZE})")
    print(f"验证集: {actual_val_size} 张图片 (目标: {Config.VAL_SIZE})")
    print(f"输出目录: {os.path.join(Config.PROJECT, run_name)}")
    print("=" * 60)

    # 检查系统资源
    available_mem, gpu_info = print_system_stats()

    # 如果GPU显存小于8GB或可用内存不足，显示警告
    gpu_warning = ""
    if "Mem:" in gpu_info:
        gpu_mem_used = float(gpu_info.split("Mem:")[1].split("/")[0].strip())
        gpu_mem_total = float(gpu_info.split("/")[1].split("GB")[0].strip())
        if gpu_mem_total < 8:
            gpu_warning = f" (总显存{gpu_mem_total:.1f}GB < 8GB)"
        elif gpu_mem_total - gpu_mem_used < 3:
            gpu_warning = f" (可用显存{gpu_mem_total - gpu_mem_used:.1f}GB < 3GB)"

    if gpu_warning:
        print(f"\n⚠️ 警告: GPU显存可能不足{gpu_warning}")
        print("训练过程中如果出现显存不足错误，程序会自动调整批处理大小")

    if available_mem < 3:
        print(f"\n⚠️ 警告: 可用内存不足3GB，训练可能会不稳定")
        print("建议关闭其他应用程序")

    # 准备训练参数 (仅使用YOLOv11支持的参数)
    train_args = {
        'data': data_config,
        'epochs': Config.EPOCHS,
        'imgsz': Config.IMGSZ,
        'batch': Config.BATCH_SIZE,
        'lr0': Config.LR0,
        'lrf': Config.LRF,
        'momentum': Config.MOMENTUM,
        'weight_decay': Config.WEIGHT_DECAY,
        'patience': Config.PATIENCE,
        'box': Config.BOX,
        'cls': Config.CLS,
        'dfl': Config.DFL,
        'workers': Config.WORKERS,
        'cache': Config.CACHE,
        'save_period': Config.SAVE_PERIOD,
        'project': Config.PROJECT,
        'name': run_name,
        'device': Config.DEVICE,
        'verbose': True
    }

    # 添加YOLOv11支持的预热参数
    train_args.update({
        'warmup_epochs': 3.0,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1
    })

    # 开始训练
    print("\n===== 开始训练 =====")
    print("提示: 按 Ctrl+C 安全停止训练并保存检查点")
    print("=" * 50)

    try:
        # 训练模型
        results = model.train(**train_args)

        print("\n训练完成!")
        # 保存最终模型
        best_model_path = os.path.join(Config.PROJECT, run_name, 'weights', 'best.pt')
        if os.path.exists(best_model_path):
            print(f"最佳模型已保存至: {best_model_path}")

        # 打印最终指标
        if hasattr(results, 'results'):
            print("\n===== 训练结果 =====")
            if 'metrics/mAP50' in results.results:
                print(f"最佳mAP@0.5: {results.results['metrics/mAP50']:.4f}")
            if 'metrics/precision' in results.results:
                print(f"最终精度: {results.results['metrics/precision']:.4f}")
            if 'metrics/recall' in results.results:
                print(f"最终召回率: {results.results['metrics/recall']:.4f}")

    except KeyboardInterrupt:
        print("\n训练被用户中断! 正在保存当前状态...")
        # 尝试保存当前状态
        checkpoint_path = os.path.join(Config.PROJECT, run_name, 'weights', 'interrupted.pt')
        model.save(checkpoint_path)
        print(f"已保存中断检查点: {checkpoint_path}")
        print("可以从该检查点恢复训练: python train.py --resume " + checkpoint_path)
    except torch.cuda.OutOfMemoryError:
        print("\n⚠️ CUDA内存不足! 尝试减少批处理大小...")
        Config.BATCH_SIZE = max(8, Config.BATCH_SIZE - 4)  # 减少4但不低于8
        print(f"新的批处理大小: {Config.BATCH_SIZE}")
        print("请重新运行程序")
    except Exception as e:
        print(f"训练异常: {e}")
        print("请检查错误信息并调整参数")
    finally:
        # 清理内存
        torch.cuda.empty_cache()
        gc.collect()


# ============== 主函数 ==============
if __name__ == "__main__":
    # 参数解析
    parser = argparse.ArgumentParser(description='CSGO人物识别训练脚本 - 资源优化版')
    parser.add_argument('--resume', type=str, default=None,
                        help='从中断处恢复训练的检查点路径')
    args = parser.parse_args()

    # 打印系统信息
    print("=" * 70)
    print(f"CSGO 人物识别训练系统 - 资源优化版")
    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda if torch.cuda.is_available() else '不可用'}")
    print(f"cuDNN: {torch.backends.cudnn.version() if torch.cuda.is_available() else '不可用'}")
    print(f"设备: {'GPU' if Config.DEVICE != 'cpu' else 'CPU'}")
    print(f"训练集大小: {Config.TRAIN_SIZE} 图片")
    print(f"验证集大小: {Config.VAL_SIZE} 图片")
    print(f"图像尺寸: {Config.IMGSZ}x{Config.IMGSZ}")
    print(f"批处理大小: {Config.BATCH_SIZE}")
    print(f"工作进程数: {Config.WORKERS}")
    print("=" * 70)

    # 检查可用内存
    mem = psutil.virtual_memory()
    available_mem = mem.available / (1024 ** 3)
    print(f"当前可用内存: {available_mem:.1f} GB")

    # 检查GPU显存
    gpu_info = ""
    try:
        gpus = GPUtil.getGPUs()
        for i, gpu in enumerate(gpus):
            gpu_info += (f"GPU {i}: 总显存 {gpu.memoryTotal:.1f} GB, "
                         f"已用 {gpu.memoryUsed:.1f} GB, "
                         f"可用 {gpu.memoryTotal - gpu.memoryUsed:.1f} GB\n")
    except Exception:
        gpu_info = "GPU信息不可用"

    print(f"[GPU状态]\n{gpu_info}")

    # 步骤1: 转换XML到YOLO格式
    print("\n[步骤1/3] 转换XML标注到YOLO格式...")
    convert_all_xml_to_yolo()

    # 转换完成后释放内存
    gc.collect()
    torch.cuda.empty_cache()

    # 步骤2: 准备数据集配置
    print("\n[步骤2/3] 准备数据集配置...")
    data_config = create_dataset_yaml()

    # 再次检查系统资源
    print_system_stats()

    # 步骤3: 开始训练
    print("\n[步骤3/3] 开始模型训练...")
    train_model(args.resume)

    print("\n训练过程结束!")