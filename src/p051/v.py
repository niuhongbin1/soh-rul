import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_and_validate_model(model_path: str, data_files: list):
    """
    加载训练好的GRU模型并进行验证
    
    Args:
        model_path: 模型文件路径
        data_files: 数据文件路径列表
    """
    
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        logger.error(f"模型文件不存在: {model_path}")
        return
    
    # 加载数据
    sequences = []
    for file_path in data_files:
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, header=0)
            if df.shape[1] >= 2:
                seq_data = df.iloc[:, :2].values.astype(float)
                seq_data = seq_data[~np.any(np.isnan(seq_data), axis=1)]
                if len(seq_data) > 60:
                    sequences.append(seq_data)
                    logger.info(f"加载数据: {file_path}, 长度: {len(seq_data)}")
    
    if len(sequences) < 3:
        logger.error("需要至少3个数据序列进行验证")
        return
    
    # 导入必要的类
    from train import GRUPredictor

    # 创建预测器并加载模型
    predictor = GRUPredictor()
    
    try:
        # 加载模型
        predictor.load_model(model_path)
        logger.info("模型加载成功!")
        
        # 使用第三个序列进行验证
        val_sequence = sequences[2]
        initial_data = val_sequence[:60]
        actual_future = val_sequence[60:, 1]
        time_future = val_sequence[60:, 0]
        
        logger.info(f"验证序列信息: 总长度={len(val_sequence)}, 预测长度={len(actual_future)}")
        
        # 进行预测
        predictions = predictor.rolling_predict(initial_data, len(actual_future))
        
        # 计算指标
        mse = mean_squared_error(actual_future, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actual_future, predictions)
        
        # 可视化结果
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 1, 1)
        plt.plot(val_sequence[:, 0], val_sequence[:, 1], 'b-', label='完整序列', linewidth=2)
        plt.plot(time_future, predictions, 'r--', label='预测值', linewidth=2)
        plt.axvline(x=time_future[0], color='gray', linestyle=':', label='预测起点')
        plt.legend()
        plt.title('GRU模型预测结果')
        plt.ylabel('数值')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 1, 2)
        plt.plot(time_future, actual_future, 'b-', label='实际值', linewidth=2)
        plt.plot(time_future, predictions, 'r--', label='预测值', linewidth=2)
        plt.legend()
        plt.xlabel('时间步')
        plt.ylabel('数值')
        plt.title('预测部分详细对比')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # 打印评估结果
        print(f"\n{'='*50}")
        print(f"{'GRU模型验证结果':^50}")
        print(f"{'='*50}")
        print(f"均方根误差 (RMSE): {rmse:.6f}")
        print(f"平均绝对误差 (MAE): {mae:.6f}")
        print(f"预测范围: [{predictions.min():.4f}, {predictions.max():.4f}]")
        print(f"实际范围: [{actual_future.min():.4f}, {actual_future.max():.4f}]")
        print(f"最终点预测: {predictions[-1]:.4f}")
        print(f"最终点实际: {actual_future[-1]:.4f}")
        print(f"最终点误差: {abs(predictions[-1] - actual_future[-1]):.4f}")
        
        return predictions, actual_future
        
    except Exception as e:
        logger.error(f"模型验证失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None

# 使用示例
if __name__ == "__main__":
    # 配置数据文件路径
    data_files = [
        './src/p051/data_parse/cleaned_5_max_1.csv',
        './src/p051/data_parse/cleaned_5_max_2.csv',
        './src/p051/data_parse/cleaned_5_max_3.csv'
    ]
    
    # 模型文件路径
    model_path = 'gru_predictor.pth'  # 或 'best_gru_model.pth'
    
    # 加载并验证模型
    predictions, actual = load_and_validate_model(model_path, data_files)