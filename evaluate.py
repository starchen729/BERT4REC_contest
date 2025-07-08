import argparse
import torch
import os
import json
from datetime import date
from types import SimpleNamespace # 用于将字典转换为可访问属性的对象

# 从现有项目导入必要的模块
# 我们不再直接从 options 导入 args，而是手动构建一个 args 对象
# from options import args
from models import model_factory
from dataloaders import dataloader_factory
from trainers import trainer_factory
from utils import * # For setup_train (to get export_root) and fix_random_seed_as

# --- 步骤 1: 定义一个最小的 ArgumentParser，只用于评估特有的参数 ---
eval_parser = argparse.ArgumentParser(description='BERT4Rec Evaluation Script')

eval_parser.add_argument('--config_path', type=str, required=True,
                         help='Path to the config.json file from the training run.')
eval_parser.add_argument('--test_model_path', type=str, required=True,
                         help='Path to the trained model checkpoint (.pth file).')
eval_parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'],
                         help='Device to use for evaluation (cpu or cuda).')

# 评估结果的输出目录
eval_parser.add_argument('--export_dir', type=str, default='evaluation_results',
                         help='Directory to save evaluation results.')

eval_args = eval_parser.parse_args()

# --- 步骤 2: 从 config.json 加载训练参数 ---
print(f"正在从 {eval_args.config_path} 加载训练配置...")
try:
    with open(eval_args.config_path, 'r') as f:
        train_config_dict = json.load(f)
    
    # 将字典转换为一个Namespace对象，使其属性可以通过点访问
    # 这将作为传递给 model_factory, dataloader_factory等的 args 对象
    full_args = SimpleNamespace(**train_config_dict)
    
    # 覆盖掉 config.json 中与评估执行相关的参数，确保使用最新的设备和模型路径
    full_args.device = eval_args.device
    full_args.test_model_path = eval_args.test_model_path # 虽然在eval_args里有，但trainer可能需要
    full_args.mode = 'test' # 确保 mode 是 test，如果trainer_factory需要
    
    # 确保 dataloader_factory 调用的参数存在 (比如负采样种子)
    # 它们应该已经在 config.json 里了，但以防万一检查一下
    if not hasattr(full_args, 'dataloader_random_seed'):
        full_args.dataloader_random_seed = 0.0 # 默认值
    if not hasattr(full_args, 'train_negative_sampling_seed'):
        full_args.train_negative_sampling_seed = 0 # 默认值，虽然评估不直接用
    if not hasattr(full_args, 'test_negative_sampling_seed'):
        full_args.test_negative_sampling_seed = 0 # 默认值

    # dataloader_factory 期望 train_batch_size, val_batch_size, test_batch_size
    # 确保 test_batch_size 有值，如果 config.json 里没有，则使用我们命令行传入的
    if not hasattr(full_args, 'test_batch_size'):
        full_args.test_batch_size = eval_parser.get_default('test_batch_size') # 获取命令行参数的默认值
    if not hasattr(full_args, 'train_batch_size'): # 即使评估用不到，dataloader_factory可能也需要
        full_args.train_batch_size = full_args.test_batch_size 
    if not hasattr(full_args, 'val_batch_size'):
        full_args.val_batch_size = full_args.test_batch_size

    # trainer_factory 期望 trainer_code
    if not hasattr(full_args, 'trainer_code'):
        full_args.trainer_code = 'bert' # 假设 bert trainer 是默认的

    print("训练配置已加载。")

except FileNotFoundError:
    print(f"错误：找不到配置文件 {eval_args.config_path}。请确保路径正确。")
    exit(1)
except json.JSONDecodeError:
    print(f"错误：配置文件 {eval_args.config_path} 不是有效的 JSON 格式。")
    exit(1)

# --- 步骤 3: 为评估结果创建输出目录 ---
eval_export_root = os.path.join(eval_args.export_dir,
                                f"eval_{os.path.basename(eval_args.test_model_path).replace('.pth', '')}_{date.today().strftime('%Y-%m-%d')}")
if not os.path.exists(eval_export_root):
    os.makedirs(eval_export_root)
print(f"评估结果将保存到: {eval_export_root}")

# --- 步骤 4: 加载数据集和数据加载器 (只需要 test_loader) ---
print("加载数据集和数据加载器...")
# dataloader_factory 将使用 full_args 中的参数
_, _, test_loader = dataloader_factory(full_args)
print("数据集和数据加载器已加载。")


# --- 步骤 5: 加载模型 ---
print(f"正在从 {eval_args.test_model_path} 加载模型...")
# model_factory 将使用 full_args 中从 config.json 读取的参数
model = model_factory(full_args) 

# 加载状态字典
# STATE_DICT_KEY 从 config.py 导入 (假设 config.py 在评估脚本中可见)
# 确保 config.py (包含 STATE_DICT_KEY) 在 evaluate.py 中可以被正确导入
# 如果 evaluate.py 不在根目录，可能需要调整 import 路径
try:
    from config import STATE_DICT_KEY
except ImportError:
    print("错误: 无法从 config.py 导入 STATE_DICT_KEY。请确保 config.py 位于可访问的路径。")
    # 如果 config.py 无法导入，你可能需要手动定义 STATE_DICT_KEY 的值
    STATE_DICT_KEY = 'model_state_dict' # 硬编码默认值，但最好还是能导入

checkpoint = torch.load(eval_args.test_model_path, map_location=eval_args.device)
model.load_state_dict(checkpoint[STATE_DICT_KEY])
model.to(eval_args.device)
model.eval() # 将模型设置为评估模式
print("模型加载成功并已设置为评估模式。")


# --- 步骤 6: 初始化 Trainer 以进行评估逻辑 ---
# trainer_factory 期望 train_loader, val_loader，所以我们传递 None
dummy_train_loader = None
dummy_val_loader = None
trainer = trainer_factory(full_args, model, dummy_train_loader, dummy_val_loader, test_loader, eval_export_root)


# --- 步骤 7: 执行评估 ---
print("正在测试集上开始评估...")
with torch.no_grad(): # 重要：评估期间不计算梯度
    # 调用 utils.py 中的 test_with 函数
    # 传入 model, test_loader, metric_ks (从 full_args 获取), device (从 full_args 获取)
    test_result = test_with(model, test_loader, full_args.metric_ks, full_args.device) # <-- 修改为这一行
print("评估完成。")


# --- 步骤 8: 保存结果 ---
save_test_result(eval_export_root, test_result)
print(f"测试结果已保存到 {eval_export_root}/test_result.json")
print("测试结果：", test_result)