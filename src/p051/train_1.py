import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

# 检查XPU可用性
if hasattr(torch, 'xpu') and torch.xpu.is_available():
    device = torch.device('xpu')
    print(f"Using XPU: {torch.xpu.get_device_name(0)}")
else:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"XPU not available, using {device}")

import pandas as pd
import numpy as np

def load_data_from_csv():
    """
    从三个CSV文件中读取序列数据
    假设每个CSV文件有两列：第一列是特征，第二列是状态值
    """
    # 读取三个CSV文件
    df1 = pd.read_csv('./src/p051/data_parse/cleaned_5_max_1.csv')  # 请替换为您的实际文件路径
    df2 = pd.read_csv('./src/p051/data_parse/cleaned_5_max_2.csv')  # 请替换为您的实际文件路径
    df3 = pd.read_csv('./src/p051/data_parse/cleaned_5_max_3.csv')  # 请替换为您的实际文件路径
    
    # 提取特征和状态值
    feature1 = df1.iloc[:, 0].values  # 第一列作为特征
    state1 = df1.iloc[:, 1].values    # 第二列作为状态值
    
    feature2 = df2.iloc[:, 0].values
    state2 = df2.iloc[:, 1].values
    
    feature3 = df3.iloc[:, 0].values
    state3 = df3.iloc[:, 1].values
    
    print(f"序列1长度: {len(feature1)}")
    print(f"序列2长度: {len(feature2)}")
    print(f"序列3长度: {len(feature3)}")
    
    return (feature1, state1), (feature2, state2), (feature3, state3)

# 替换原来的generate_sample_data函数
def generate_sample_data():
    return load_data_from_csv()

# 自定义数据集类
class SequenceDataset(Dataset):
    def __init__(self, sequences):
        """
        sequences: list of tuples [(features1, states1), (features2, states2), ...]
        """
        self.features = []
        self.states = []
        
        for features, states in sequences:
            self.features.extend(features)
            self.states.extend(states)
        
        self.features = torch.FloatTensor(self.features).unsqueeze(1)  # 添加特征维度
        self.states = torch.FloatTensor(self.states).unsqueeze(1)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.states[idx]

# MLP模型定义
class MLPRegressor(nn.Module):
    def __init__(self, input_dim=1, hidden_dims=[64, 128, 64], output_dim=1, dropout=0.1):
        super(MLPRegressor, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

# 训练函数
def train_model(model, train_loader, val_loader, num_epochs=1000, learning_rate=0.001):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=50, factor=0.5)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        for features, states in train_loader:
            features, states = features.to(device), states.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, states)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for features, states in val_loader:
                features, states = features.to(device), states.to(device)
                outputs = model(features)
                loss = criterion(outputs, states)
                val_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        scheduler.step(avg_val_loss)
        
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}')
    
    return train_losses, val_losses

# 预测函数
def predict_sequence(model, features):
    model.eval()
    with torch.no_grad():
        features_tensor = torch.FloatTensor(features).unsqueeze(1).to(device)
        predictions = model(features_tensor)
        return predictions.cpu().numpy().flatten()

# 主程序
def main():
    # 生成示例数据
    seq1, seq2, seq3 = generate_sample_data()
    
    print(f"序列1长度: {len(seq1[0])}")
    print(f"序列2长度: {len(seq2[0])}")
    print(f"序列3长度: {len(seq3[0])}")
    
    # 准备训练数据（使用序列1和序列2）
    train_sequences = [seq1, seq2]
    dataset = SequenceDataset(train_sequences)
    
    # 分割训练集和验证集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # 创建模型
    model = MLPRegressor(input_dim=1, hidden_dims=[128, 256, 128], output_dim=1, dropout=0.2)
    model.to(device)
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")
    
    # 训练模型
    print("开始训练模型...")
    train_losses, val_losses = train_model(
        model, train_loader, val_loader, 
        num_epochs=1000, learning_rate=0.001
    )
    
    # 对第三条序列进行预测
    print("对第三条序列进行预测...")
    features3, true_states3 = seq3
    predictions3 = predict_sequence(model, features3)
    
    # 计算预测误差
    mse = np.mean((predictions3 - true_states3) ** 2)
    mae = np.mean(np.abs(predictions3 - true_states3))
    print(f"预测误差 - MSE: {mse:.6f}, MAE: {mae:.6f}")
    
    # 可视化结果
    plt.figure(figsize=(15, 5))
    
    # 训练损失
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training History')
    plt.yscale('log')
    
    # 序列1和2的拟合情况
    plt.subplot(1, 3, 2)
    # 显示序列1
    features1, true_states1 = seq1
    pred1 = predict_sequence(model, features1)
    plt.plot(features1, true_states1, 'b-', alpha=0.7, label='Seq1 True')
    plt.plot(features1, pred1, 'r--', alpha=0.8, label='Seq1 Pred')
    
    # 显示序列2
    features2, true_states2 = seq2
    pred2 = predict_sequence(model, features2)
    plt.plot(features2, true_states2, 'g-', alpha=0.7, label='Seq2 True')
    plt.plot(features2, pred2, 'm--', alpha=0.8, label='Seq2 Pred')
    
    plt.xlabel('Feature')
    plt.ylabel('State')
    plt.legend()
    plt.title('Training Sequences Fit')
    
    # 序列3的预测结果
    plt.subplot(1, 3, 3)
    plt.plot(features3, true_states3, 'b-', linewidth=2, label='True States')
    plt.plot(features3, predictions3, 'r--', linewidth=2, label='Predictions')
    plt.xlabel('Feature')
    plt.ylabel('State')
    plt.legend()
    plt.title(f'Sequence 3 Prediction\nMSE: {mse:.6f}')
    
    plt.tight_layout()
    plt.show()
    
    # 保存模型
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_architecture': model.network
    }, 'mlp_sequence_regressor.pth')
    print("模型已保存为 'mlp_sequence_regressor.pth'")

if __name__ == "__main__":
    main()