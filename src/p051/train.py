import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Tuple, Optional, Dict, Any
import warnings
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
from torch.utils.data import Dataset, DataLoader
import logging

warnings.filterwarnings('ignore')

# 设置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 设置中文字体显示
def set_chinese_font():
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        print("中文字体设置成功")
    except:
        print("警告: 中文字体设置失败，使用默认字体")

set_chinese_font()

# 检测可用设备
def get_device():
    if torch.xpu.is_available():
        device = torch.device("xpu")
        print(f"使用 Intel XPU: {torch.xpu.get_device_name(0)}")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("使用 NVIDIA GPU")
    else:
        device = torch.device("cpu")
        print("使用 CPU")
    return device

device = get_device()

class RobustFeatureEngine:
    """简化的特征工程类 - 只进行序列标准化，不提取统计特征"""
    
    def __init__(self):
        self.scaler_time = StandardScaler()  # 时间列标准化器
        self.scaler_value = StandardScaler()  # 数值列标准化器
        self.scaler_y = StandardScaler()  # 目标值标准化器
        self.is_fitted = False
    
    def fit(self, train_sequences: List[np.ndarray]):
        """在训练数据上拟合标准化器 - 对整个序列进行标准化"""
        all_time_data = []
        all_value_data = []
        all_targets = []
        
        # 收集所有时间数据和数值数据
        for seq in train_sequences:
            if len(seq) < 60:
                continue
                
            time_data = seq[:, 0].reshape(-1, 1)  # 时间列
            value_data = seq[:, 1].reshape(-1, 1)  # 数值列
            
            all_time_data.append(time_data)
            all_value_data.append(value_data)
            
            # 收集目标值（后60个点的数值列）
            for start_idx in range(0, len(seq) - 120 + 1, 10):
                window = seq[start_idx:start_idx+120]
                y_data = window[60:, 1]  # 后60个点的第二列作为目标
                all_targets.extend(y_data)
        
        if len(all_time_data) > 0 and len(all_value_data) > 0:
            # 拟合标准化器
            all_time_array = np.vstack(all_time_data)
            all_value_array = np.vstack(all_value_data)
            
            self.scaler_time.fit(all_time_array)
            self.scaler_value.fit(all_value_array)
            
            if len(all_targets) > 0:
                all_targets_array = np.array(all_targets).reshape(-1, 1)
                self.scaler_y.fit(all_targets_array)
            
            self.is_fitted = True
            logger.info(f"序列标准化器拟合完成 - 时间维度: {all_time_array.shape}, 数值维度: {all_value_array.shape}")
        else:
            logger.warning("没有足够的训练数据来拟合标准化器")
    
    def transform_sequence(self, sequence: np.ndarray) -> np.ndarray:
        """对整个序列进行标准化"""
        if not self.is_fitted:
            return sequence
        
        time_normalized = self.scaler_time.transform(sequence[:, 0].reshape(-1, 1)).flatten()
        value_normalized = self.scaler_value.transform(sequence[:, 1].reshape(-1, 1)).flatten()
        
        return np.column_stack([time_normalized, value_normalized])
    
    def inverse_transform_value(self, value_normalized: np.ndarray) -> np.ndarray:
        """反标准化数值列"""
        if not self.is_fitted:
            return value_normalized
        
        return self.scaler_value.inverse_transform(value_normalized.reshape(-1, 1)).flatten()
    
    def inverse_transform_targets(self, y_data: np.ndarray) -> np.ndarray:
        """反转换目标（应用反标准化）"""
        if not self.is_fitted:
            return y_data
        
        return self.scaler_y.inverse_transform(y_data.reshape(-1, 1)).flatten()
    
    def transform_features(self, x_data: np.ndarray) -> np.ndarray:
        """转换特征 - 直接返回标准化后的两列数据"""
        return x_data  # 已经标准化过了，直接返回
    
    def transform_targets(self, y_data: np.ndarray) -> np.ndarray:
        """转换目标（应用标准化）"""
        if not self.is_fitted:
            return y_data
        
        return self.scaler_y.transform(y_data.reshape(-1, 1)).flatten()
    
    def get_state(self) -> Dict[str, Any]:
        """获取特征工程的状态（用于保存）"""
        return {
            'scaler_time_mean': self.scaler_time.mean_,
            'scaler_time_scale': self.scaler_time.scale_,
            'scaler_value_mean': self.scaler_value.mean_,
            'scaler_value_scale': self.scaler_value.scale_,
            'scaler_y_mean': self.scaler_y.mean_,
            'scaler_y_scale': self.scaler_y.scale_,
            'is_fitted': self.is_fitted
        }
    
    def set_state(self, state: Dict[str, Any]):
        """设置特征工程的状态（用于加载）"""
        self.scaler_time.mean_ = state['scaler_time_mean']
        self.scaler_time.scale_ = state['scaler_time_scale']
        self.scaler_value.mean_ = state['scaler_value_mean']
        self.scaler_value.scale_ = state['scaler_value_scale']
        self.scaler_y.mean_ = state['scaler_y_mean']
        self.scaler_y.scale_ = state['scaler_y_scale']
        
        self.scaler_time.n_features_in_ = len(state['scaler_time_mean'])
        self.scaler_value.n_features_in_ = len(state['scaler_value_mean'])
        self.scaler_y.n_features_in_ = len(state['scaler_y_mean'])
        self.is_fitted = state['is_fitted']

class GRUSlidingWindowDataset(Dataset):
    """GRU滑动窗口数据集 - 使用标准化后的两列数据直接作为输入"""
    
    def __init__(self, sequences: List[np.ndarray], feature_engine: RobustFeatureEngine,
                 window_size: int = 120, stride: int = 10, 
                 input_length: int = 60, output_length: int = 60,
                 mode: str = 'train'):
        self.sequences = sequences
        self.feature_engine = feature_engine
        self.window_size = window_size
        self.stride = stride
        self.input_length = input_length
        self.output_length = output_length
        self.mode = mode
        
        # 首先对所有序列进行标准化
        self.normalized_sequences = []
        for seq in sequences:
            normalized_seq = self.feature_engine.transform_sequence(seq)
            self.normalized_sequences.append(normalized_seq)
        
        self.samples = self._create_samples()
    
    def _create_samples(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        samples = []
        
        for norm_seq in self.normalized_sequences:
            seq_len = len(norm_seq)
            
            if seq_len < self.window_size:
                logger.warning(f"序列长度 {seq_len} 小于窗口大小 {self.window_size}，跳过该序列")
                continue
            
            # 正向滑动窗口
            for start_idx in range(0, seq_len - self.window_size + 1, self.stride):
                end_idx = start_idx + self.window_size
                window = norm_seq[start_idx:end_idx]
                
                # 提取输入和输出
                x_data = window[:self.input_length]  # 前60个点 (60, 2)
                y_data = window[self.input_length:, 1]  # 后60个点的第二列（标准化后的数值）
                
                # 直接使用标准化后的两列数据作为输入
                try:
                    # 这里不再进行特征提取，直接使用标准化后的数据
                    x_enhanced = self.feature_engine.transform_features(x_data)  # 返回 (60, 2)
                    y_normalized = self.feature_engine.transform_targets(y_data)  # 返回 (60,)
                    
                    samples.append((x_enhanced, y_normalized))
                except Exception as e:
                    logger.warning(f"数据转换失败，跳过样本: {e}")
                    continue
            
            # 末尾逆向窗口
            if seq_len >= self.window_size:
                last_window = norm_seq[-self.window_size:]
            else:
                # 如果序列长度不足，用第一个元素填充
                padding = np.full((self.window_size - seq_len, 2), norm_seq[0])
                last_window = np.vstack([padding, norm_seq])
            
            # 提取输入和输出
            x_data = last_window[:self.input_length]
            y_data = last_window[self.input_length:, 1]
            
            # 直接使用标准化后的两列数据作为输入
            try:
                x_enhanced = self.feature_engine.transform_features(x_data)
                y_normalized = self.feature_engine.transform_targets(y_data)
                
                samples.append((x_enhanced, y_normalized))
            except Exception as e:
                logger.warning(f"数据转换失败，跳过样本: {e}")
                continue
        
        logger.info(f"创建了 {len(samples)} 个样本")
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.FloatTensor(x), torch.FloatTensor(y)

class GRUSequencePredictor(nn.Module):
    """GRU序列预测模型 - 输入维度修改为2"""
    
    def __init__(self, input_size: int = 2, hidden_size: int = 256, 
                 num_layers: int = 3, output_length: int = 60, dropout: float = 0.3):
        super(GRUSequencePredictor, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_length = output_length
        
        # GRU层
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        
        # 全连接层
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_size * 2, 512),  # 双向GRU，所以是hidden_size * 2
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_length)
        )
        
        # 权重初始化
        self._initialize_weights()
    
    def _initialize_weights(self):
        for name, param in self.gru.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
        
        for module in self.fc_layers.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, features) - features现在为2
        
        # GRU层
        gru_out, _ = self.gru(x)  # (batch, seq_len, hidden_size * 2)
        
        # 取最后一个时间步的输出
        last_output = gru_out[:, -1, :]  # (batch, hidden_size * 2)
        
        # 全连接层
        output = self.fc_layers(last_output)  # (batch, output_length)
        
        return output

class GRUPredictor:
    """GRU预测器 - 使用简化的两列输入"""
    
    def __init__(self, input_length: int = 60, output_length: int = 60,
                 hidden_size: int = 256, num_layers: int = 3,
                 learning_rate: float = 0.001, dropout: float = 0.3):
        self.input_length = input_length
        self.output_length = output_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.model = None
        self.device = device
        self.feature_engine = RobustFeatureEngine()
        
    def prepare_data(self, train_sequences: List[np.ndarray], 
                    val_sequence: np.ndarray) -> Tuple[DataLoader, DataLoader]:
        """准备数据"""
        # 首先在训练数据上拟合特征工程（对整个序列进行标准化）
        logger.info("在训练数据上拟合序列标准化器...")
        self.feature_engine.fit(train_sequences)
        
        # 创建数据集（内部会对序列进行标准化）
        train_dataset = GRUSlidingWindowDataset(
            train_sequences, 
            self.feature_engine,
            window_size=120,  # 60输入 + 60输出
            stride=10,
            input_length=self.input_length,
            output_length=self.output_length,
            mode='train'
        )
        
        val_dataset = GRUSlidingWindowDataset(
            [val_sequence],
            self.feature_engine,
            window_size=120,
            stride=10,
            input_length=self.input_length,
            output_length=self.output_length,
            mode='val'
        )
        
        logger.info(f"训练样本数: {len(train_dataset)}")
        logger.info(f"验证样本数: {len(val_dataset)}")
        
        # 获取输入特征维度
        sample_x, _ = train_dataset[0]
        input_size = sample_x.shape[1]  # 特征维度，现在应该是2
        logger.info(f"输入特征维度: {input_size}")
        
        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
        
        return train_loader, val_loader, input_size
    
    def train(self, train_sequences: List[np.ndarray], val_sequence: np.ndarray,
              epochs: int = 200, patience: int = 30):
        """训练模型"""
        logger.info("开始准备数据...")
        train_loader, val_loader, input_size = self.prepare_data(train_sequences, val_sequence)
        
        # 创建模型 - 输入维度现在固定为2
        self.model = GRUSequencePredictor(
            input_size=input_size,  # 现在固定为2
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            output_length=self.output_length,
            dropout=self.dropout
        ).to(self.device)
        
        # 损失函数和优化器
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate, 
                               weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        logger.info("开始训练GRU模型...")
        logger.info(f"模型参数量: {sum(p.numel() for p in self.model.parameters())}")
        
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # 训练阶段
            self.model.train()
            epoch_train_loss = 0
            for x_batch, y_batch in train_loader:
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                
                optimizer.zero_grad()
                predictions = self.model(x_batch)
                loss = criterion(predictions, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                epoch_train_loss += loss.item()
            
            avg_train_loss = epoch_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            # 验证阶段
            self.model.eval()
            epoch_val_loss = 0
            with torch.no_grad():
                for x_val, y_val in val_loader:
                    x_val, y_val = x_val.to(self.device), y_val.to(self.device)
                    val_predictions = self.model(x_val)
                    val_loss = criterion(val_predictions, y_val)
                    epoch_val_loss += val_loss.item()
            
            avg_val_loss = epoch_val_loss / len(val_loader)
            val_losses.append(avg_val_loss)
            
            # 学习率调整
            scheduler.step(avg_val_loss)
            
            # 早停检查
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # 保存模型状态而不是整个对象
                self._save_model('best_gru_model.pth')
                logger.info(f"保存最佳模型，验证损失: {best_val_loss:.6f}")
            else:
                patience_counter += 1
            
            # 打印进度
            if epoch % 20 == 0 or epoch == epochs - 1:
                current_lr = optimizer.param_groups[0]['lr']
                logger.info(f'Epoch [{epoch}/{epochs}], Train Loss: {avg_train_loss:.6f}, '
                      f'Val Loss: {avg_val_loss:.6f}, LR: {current_lr:.6f}, '
                      f'Patience: {patience_counter}/{patience}')
            
            # 早停
            if patience_counter >= patience:
                logger.info(f"早停触发于第 {epoch} 轮")
                break
        
        # 加载最佳模型
        if os.path.exists('best_gru_model.pth'):
            self._load_model('best_gru_model.pth')
        
        logger.info(f"训练完成! 最佳验证损失: {best_val_loss:.6f}")
        return train_losses, val_losses
    
    def _save_model(self, filepath: str):
        """保存模型状态"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'feature_engine_state': self.feature_engine.get_state(),
            'config': {
                'input_length': self.input_length,
                'output_length': self.output_length,
                'hidden_size': self.hidden_size,
                'num_layers': self.num_layers,
                'learning_rate': self.learning_rate,
                'dropout': self.dropout
            }
        }
        torch.save(checkpoint, filepath)
    
    def _load_model(self, filepath: str):
        """加载模型状态"""
        try:
            # 首先尝试使用 weights_only=True
            checkpoint = torch.load(filepath, weights_only=False)
        except Exception as e:
            logger.warning(f"使用 weights_only=False 加载失败: {e}")
            # 如果失败，使用传统方式加载
            checkpoint = torch.load(filepath, map_location=self.device)
        
        if self.model is None:
            # 如果模型未初始化，需要先创建模型
            self.model = GRUSequencePredictor(
                input_size=2,  # 现在固定为2
                hidden_size=checkpoint['config']['hidden_size'],
                num_layers=checkpoint['config']['num_layers'],
                output_length=checkpoint['config']['output_length'],
                dropout=checkpoint['config']['dropout']
            ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.feature_engine.set_state(checkpoint['feature_engine_state'])
        
        # 更新配置
        config = checkpoint['config']
        self.input_length = config['input_length']
        self.output_length = config['output_length']
        self.hidden_size = config['hidden_size']
        self.num_layers = config['num_layers']
        self.learning_rate = config['learning_rate']
        self.dropout = config['dropout']
    
    def rolling_predict(self, initial_sequence: np.ndarray, total_prediction_length: int) -> np.ndarray:
        """滚动预测方法"""
        if self.model is None:
            raise ValueError("模型未训练")
            
        self.model.eval()
        
        # 首先对初始序列进行标准化
        normalized_sequence = self.feature_engine.transform_sequence(initial_sequence)
        current_sequence = normalized_sequence.copy()
        
        all_predictions = []
        
        # 计算需要多少次滚动预测
        num_rolls = (total_prediction_length + 9) // 10  # 每次保留10个点
        
        logger.info(f"需要 {num_rolls} 次滚动预测，总共预测 {num_rolls * 10} 个点")
        
        with torch.no_grad():
            for roll in range(num_rolls):
                # 确保当前序列长度足够
                if len(current_sequence) < self.input_length:
                    # 填充序列
                    padding = np.full((self.input_length - len(current_sequence), 2), current_sequence[0])
                    current_input = np.vstack([padding, current_sequence])
                else:
                    current_input = current_sequence[-self.input_length:]
                
                try:
                    # 直接使用标准化后的两列数据
                    x_tensor = torch.FloatTensor(current_input).unsqueeze(0).to(self.device)
                    
                    # 预测60个点
                    predictions = self.model(x_tensor)
                    pred_values_normalized = predictions.cpu().numpy()[0]
                    
                    # 反标准化预测值
                    pred_values = self.feature_engine.inverse_transform_targets(pred_values_normalized)
                    
                    # 只保留前10个预测点
                    retained_predictions = pred_values[:10]
                    all_predictions.extend(retained_predictions)
                    
                    # 更新序列 - 使用保留的10个预测值
                    start_time = current_sequence[-1, 0] + 1 if len(current_sequence) > 0 else 1
                    for i, pred_val in enumerate(retained_predictions):
                        # 创建新的标准化点
                        new_time = start_time + i
                        # 注意：这里我们只更新数值列，时间列按顺序递增
                        new_point = np.array([[new_time, pred_val]])
                        # 对新点进行标准化（只标准化数值列）
                        new_point_normalized = self.feature_engine.transform_sequence(new_point)
                        current_sequence = np.vstack([current_sequence, new_point_normalized[0]])
                    
                    logger.info(f"滚动预测第 {roll+1}/{num_rolls} 次完成, 保留 {len(retained_predictions)} 个点, 当前序列长度: {len(current_sequence)}")
                
                except Exception as e:
                    logger.error(f"滚动预测第 {roll+1} 次失败: {e}")
                    # 在失败时使用简单的策略：使用最后一个值作为预测
                    last_value = current_sequence[-1, 1]
                    retained_predictions = [last_value] * 10
                    all_predictions.extend(retained_predictions)
                    
                    # 更新序列
                    start_time = current_sequence[-1, 0] + 1
                    for i, pred_val in enumerate(retained_predictions):
                        new_point = np.array([[start_time + i, pred_val]])
                        new_point_normalized = self.feature_engine.transform_sequence(new_point)
                        current_sequence = np.vstack([current_sequence, new_point_normalized[0]])
        
        # 只返回需要的长度
        return np.array(all_predictions[:total_prediction_length])
    
    def predict_on_train(self, train_sequences: List[np.ndarray]) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """在训练集上进行预测 - 返回时序信息用于绘图"""
        self.model.eval()
        results = []
        
        for seq_idx, seq in enumerate(train_sequences):
            if len(seq) <= self.input_length:
                continue
                
            # 使用前60个点预测后面的点
            initial_data = seq[:self.input_length]
            actual_future = seq[self.input_length:, 1]
            time_future = seq[self.input_length:, 0]  # 用于绘图的时序信息
            
            # 进行滚动预测
            try:
                predictions = self.rolling_predict(initial_data, len(actual_future))
                results.append((predictions, actual_future, time_future))
            except Exception as e:
                logger.error(f"训练序列 {seq_idx+1} 预测失败: {e}")
                continue
        
        return results

# 其余函数保持不变
def load_and_preprocess_data(csv_files: List[str] = None) -> List[np.ndarray]:
    """数据加载和预处理 - 增强了错误处理"""
    if csv_files is None:
        csv_files = ['./src/p051/data_parse/cleaned_5_max_1.csv',
                     './src/p051/data_parse/cleaned_5_max_2.csv', 
                     './src/p051/data_parse/cleaned_5_max_3.csv']
    
    sequences = []
    logger.info("数据加载和分析:")
    
    for file_path in csv_files:
        try:
            if not os.path.exists(file_path):
                logger.warning(f"文件 {file_path} 不存在")
                continue
                
            df = pd.read_csv(file_path, header=0)
            
            if df.shape[1] >= 2:
                seq_data = df.iloc[:, :2].values.astype(float)
                # 移除包含NaN的行
                seq_data = seq_data[~np.any(np.isnan(seq_data), axis=1)]
                
                if len(seq_data) > 60:
                    sequences.append(seq_data)
                    
                    values = seq_data[:, 1]
                    trend = np.polyfit(np.arange(len(values)), values, 1)[0]
                    value_range = np.max(values) - np.min(values)
                    logger.info(f"文件 {file_path}: {len(seq_data)} 行数据, "
                          f"趋势斜率: {trend:.6f}, 值域: {value_range:.4f}")
                else:
                    logger.warning(f"文件 {file_path} 数据长度不足60")
            else:
                logger.warning(f"文件 {file_path} 列数不足")
                    
        except Exception as e:
            logger.error(f"读取文件 {file_path} 时发生错误: {e}")
    
    return sequences

def evaluate_gru_model(predictor, sequences):
    """评估GRU模型 - 使用时序信息作为横轴"""
    if len(sequences) < 3:
        logger.error("数据不足")
        return
    
    # 训练集预测
    logger.info("在训练集上进行预测...")
    train_results = predictor.predict_on_train(sequences[:2])
    
    # 验证集预测
    val_sequence = sequences[2]
    initial_data = val_sequence[:60]  # 前60个点作为输入
    actual_future = val_sequence[60:, 1]  # 后60个点的第二列作为实际值
    time_future = val_sequence[60:, 0]  # 后60个点的第一列作为时序信息
    
    logger.info(f"验证序列长度: {len(val_sequence)}")
    logger.info(f"初始输入长度: {len(initial_data)}")
    logger.info(f"需要预测长度: {len(actual_future)}")
    
    # 进行滚动预测
    try:
        predictions = predictor.rolling_predict(initial_data, len(actual_future))
        logger.info(f"预测值范围: [{np.min(predictions):.4f}, {np.max(predictions):.4f}]")
        logger.info(f"实际值范围: [{np.min(actual_future):.4f}, {np.max(actual_future):.4f}]")
    except Exception as e:
        logger.error(f"预测过程中出现错误: {e}")
        return None, None
    
    # 计算评估指标
    mse = mean_squared_error(actual_future, predictions)
    mae = mean_absolute_error(actual_future, predictions)
    rmse = np.sqrt(mse)
    
    # 计算趋势误差
    actual_trend = np.polyfit(np.arange(len(actual_future)), actual_future, 1)[0]
    pred_trend = np.polyfit(np.arange(len(predictions)), predictions, 1)[0]
    trend_error = abs(pred_trend - actual_trend)
    
    # 创建评估图形
    fig = plt.figure(figsize=(15, 10))
    
    # 1. 训练集表现 - 使用时序信息作为横轴
    for i, (train_pred, train_actual, train_time) in enumerate(train_results):
        plt.subplot(2, 3, i+1)
        
        plt.plot(train_time, train_actual, label='实际值', linewidth=2, color='blue')
        plt.plot(train_time[:len(train_pred)], train_pred, label='预测值', linewidth=2, color='red', linestyle='--')
        plt.xlabel('时序信息')
        plt.ylabel('数值')
        plt.legend()
        plt.title(f'训练序列 {i+1} 预测')
        plt.grid(True, alpha=0.3)
        
        # 训练集指标
        train_rmse = np.sqrt(mean_squared_error(train_actual[:len(train_pred)], train_pred))
        plt.text(0.05, 0.95, f'RMSE: {train_rmse:.4f}', transform=plt.gca().transAxes, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # 2. 验证集完整视图 - 使用时序信息作为横轴
    plt.subplot(2, 3, 3)
    full_time = val_sequence[:, 0]  # 完整的时序信息
    full_actual = val_sequence[:, 1]  # 完整的实际值
    
    plt.plot(full_time, full_actual, label='完整实际序列', linewidth=2, color='blue')
    plt.plot(time_future[:len(predictions)], predictions, label='预测序列', linewidth=2, color='red', linestyle='--')
    plt.axvline(x=time_future[0], color='gray', linestyle=':', alpha=0.7, label='预测起点')
    plt.xlabel('时序信息')
    plt.ylabel('数值')
    plt.legend()
    plt.title('验证序列 - 完整视图')
    plt.grid(True, alpha=0.3)
    
    # 3. 验证集预测部分 - 使用时序信息作为横轴
    plt.subplot(2, 3, 4)
    plt.plot(time_future, actual_future, label='实际值', linewidth=2, color='blue')
    plt.plot(time_future[:len(predictions)], predictions, label='预测值', linewidth=2, color='red', linestyle='--')
    plt.xlabel('时序信息')
    plt.ylabel('数值')
    plt.legend()
    plt.title('验证序列 - 预测部分')
    plt.grid(True, alpha=0.3)
    
    # 4. 误差分析
    plt.subplot(2, 3, 5)
    errors = np.abs(predictions - actual_future[:len(predictions)])
    plt.plot(time_future[:len(predictions)], errors, label='绝对误差', color='orange', linewidth=1)
    plt.xlabel('时序信息')
    plt.ylabel('绝对误差')
    plt.legend()
    plt.title('误差分析')
    plt.grid(True, alpha=0.3)
    
    # 5. 训练历史
    plt.subplot(2, 3, 6)
    if hasattr(predictor, 'train_losses'):
        plt.plot(predictor.train_losses, label='训练损失', color='blue')
        plt.plot(predictor.val_losses, label='验证损失', color='red')
        plt.xlabel('训练轮次')
        plt.ylabel('损失')
        plt.legend()
        plt.title('训练历史')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 打印详细统计
    print(f"\n=== GRU模型评估结果 ===")
    print(f"均方误差 (MSE): {mse:.6f}")
    print(f"均方根误差 (RMSE): {rmse:.6f}")
    print(f"平均绝对误差 (MAE): {mae:.6f}")
    print(f"预测长度: {len(predictions)} 个时间步")
    print(f"最终预测值: {predictions[-1]:.4f}")
    print(f"最终实际值: {actual_future[-1]:.4f}")
    print(f"最终误差: {abs(predictions[-1] - actual_future[-1]):.4f}")
    print(f"实际趋势斜率: {actual_trend:.6f}")
    print(f"预测趋势斜率: {pred_trend:.6f}")
    print(f"趋势捕捉误差: {trend_error:.6f}")
    
    # 训练集统计
    print(f"\n=== 训练集表现 ===")
    for i, (train_pred, train_actual, _) in enumerate(train_results):
        train_mse = mean_squared_error(train_actual[:len(train_pred)], train_pred)
        train_rmse = np.sqrt(train_mse)
        train_mae = mean_absolute_error(train_actual[:len(train_pred)], train_pred)
        print(f"训练序列 {i+1}: RMSE={train_rmse:.4f}, MAE={train_mae:.4f}")
    
    return predictions, actual_future

def train_gru_model():
    """训练GRU模型"""
    sequences = load_and_preprocess_data()
    
    if len(sequences) < 2:
        logger.error("数据不足")
        return None, sequences
    
    logger.info(f"\n数据统计:")
    logger.info(f"训练序列数量: 2 (长度: {len(sequences[0])}, {len(sequences[1])})")
    logger.info(f"验证序列长度: {len(sequences[2])}")
    
    # 创建GRU预测器
    predictor = GRUPredictor(
        input_length=60,
        output_length=60,
        hidden_size=256,
        num_layers=3,
        learning_rate=0.001,
        dropout=0.3
    )
    
    # 训练模型
    train_sequences = sequences[:2]
    val_sequence = sequences[2]
    
    try:
        train_losses, val_losses = predictor.train(
            train_sequences, 
            val_sequence,
            epochs=200,
            patience=30
        )
        
        predictor.train_losses = train_losses
        predictor.val_losses = val_losses
        
        return predictor, sequences
    except Exception as e:
        logger.error(f"训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return None, sequences

# 运行训练和评估
if __name__ == "__main__":
    logger.info("开始训练GRU序列预测模型...")
    try:
        predictor, sequences = train_gru_model()
        
        if predictor is not None:
            predictions, actual = evaluate_gru_model(predictor, sequences)
            
            if predictions is not None:
                predictor._save_model('gru_predictor.pth')
                logger.info("\nGRU模型已保存为 'gru_predictor.pth'")
        else:
            logger.error("训练失败")
            
    except Exception as e:
        logger.error(f"训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()