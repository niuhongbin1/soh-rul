import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from typing import List, Tuple
import math
import warnings
import os
warnings.filterwarnings('ignore')

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

class EnhancedSequenceDataset(Dataset):
    def __init__(self, sequences: List[np.ndarray], input_length: int = 120,
                 output_length: int = 100, stride: int = 5, mode: str = 'train'):
        self.sequences = sequences
        self.input_length = input_length
        self.output_length = output_length
        self.stride = stride
        self.mode = mode
        
        self.samples = self._create_samples()
    
    def _create_samples(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        samples = []
        
        for seq in self.sequences:
            seq_len = len(seq)
            
            if self.mode == 'train':
                # 训练模式：创建多个滑动窗口
                max_start = seq_len - self.input_length - self.output_length
                for i in range(0, max_start + 1, self.stride):
                    x = seq[i:i + self.input_length]  # 输入：所有特征
                    y = seq[i + self.input_length:i + self.input_length + self.output_length, 1:2]  # 输出：只第二列
                    samples.append((x, y))
            else:
                # 验证/测试模式：只取前input_length个点
                x = seq[:self.input_length]  # 输入：所有特征
                y = seq[self.input_length:, 1:2]  # 输出：只第二列
                samples.append((x, y))
                
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.FloatTensor(x), torch.FloatTensor(y)

# Informer模型的核心组件（简化实现，基于官方Informer）
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(1e4) / d_model)).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model, kernel_size=3, padding=1, padding_mode='circular')
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x

class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1):
        super(DataEmbedding, self).__init__()
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)

class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):
        B, H, L_Q, E = Q.shape
        _, _, L_K, _ = K.shape
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k))
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]
        Q_reduce = Q[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], M_top, :]
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))
        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        V_sum = V.mean(dim=-2)
        contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q):
        B, H, L_V, D = V.shape
        attn = torch.softmax(scores, dim=-1)
        context_in[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = torch.matmul(attn, V).type_as(context_in)
        return context_in, attn

    def forward(self, queries, keys, values, attn_mask):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape
        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)
        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item()
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item()
        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q
        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)
        scale = self.scale or 1. / math.sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        context = self._get_initial_context(values, L_Q)
        context, attn = self._update_context(context, values, scores_top, index, L_Q)
        return context.transpose(2, 1).contiguous(), attn

class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None, mix=False):
        super(AttentionLayer, self).__init__()
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.mix = mix

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)
        out, attn = self.inner_attention(queries, keys, values, attn_mask)
        if self.mix:
            out = out.transpose(2, 1).contiguous()
        out = out.view(B, L, -1)
        return self.out_projection(out), attn

class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU() if activation == "relu" else nn.GELU()

    def forward(self, x, attn_mask=None):
        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y), attn

class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        attns = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                # 添加transpose以修复Conv1d输入（如果distil=True时使用）
                x = x.transpose(1, 2)  # [B, L, D] -> [B, D, L]
                x = conv_layer(x)
                x = x.transpose(1, 2)  # back to [B, L, D]
                attns.append(attn)
            x, attn = self.attn_layers[-1](x)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)
        if self.norm is not None:
            x = self.norm(x)
        return x, attns

class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU() if activation == "relu" else nn.GELU()

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        x = x + self.dropout(self.self_attention(x, x, x, attn_mask=x_mask)[0])
        x = self.norm1(x)
        x = x + self.dropout(self.cross_attention(x, cross, cross, attn_mask=cross_mask)[0])
        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm3(x + y)

class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)
        if self.norm is not None:
            x = self.norm(x)
        return x

class Informer(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, pred_len, 
                 factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512, 
                 dropout=0.0, activation='gelu', output_attention=False, distil=True):
        super(Informer, self).__init__()
        self.pred_len = pred_len
        self.output_attention = output_attention

        # 嵌入
        self.enc_embedding = DataEmbedding(enc_in, d_model, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, dropout)

        # 编码器
        Attn = ProbAttention
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention), 
                                   d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [
                nn.Conv1d(d_model, d_model, kernel_size=3, stride=1, padding=1, groups=d_model) 
                for l in range(e_layers - 1)
            ] if distil else None,
            norm_layer=nn.LayerNorm(d_model)
        )

        # 解码器
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False), 
                                   d_model, n_heads, mix=False),
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=False), 
                                   d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=nn.LayerNorm(d_model)
        )

        self.projection = nn.Linear(d_model, c_out, bias=True)

    def forward(self, x_enc, x_dec, enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        enc_out = self.enc_embedding(x_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out = self.projection(dec_out)
        return dec_out[:, -self.pred_len:, :]  # 只返回预测部分

def calculate_trend_features(sequence: np.ndarray, window_sizes: List[int] = [10, 20, 30]) -> np.ndarray:
    """计算多尺度趋势特征"""
    features = []
    
    for window_size in window_sizes:
        trend_feature = np.zeros(len(sequence))
        
        for i in range(len(sequence)):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(sequence), i + window_size // 2 + 1)
            
            if end_idx - start_idx > 1:
                window_data = sequence[start_idx:end_idx, 1]  # 第二列
                x = np.arange(len(window_data))
                slope = np.polyfit(x, window_data, 1)[0]
                trend_feature[i] = slope
            else:
                trend_feature[i] = 0
        
        features.append(trend_feature)
    
    if features:
        trend_features = np.column_stack(features)
        enhanced_sequence = np.column_stack([sequence, trend_features])
        return enhanced_sequence
    else:
        return sequence

def robust_standardize_sequences(sequences: List[np.ndarray]) -> Tuple[List[np.ndarray], List[dict]]:
    """改进的标准化：使用训练集统计量"""
    scalers = []
    standardized_seqs = []
    
    if len(sequences) >= 2:
        train_sequences = sequences[:2]
        all_train_data = np.vstack(train_sequences)
        global_mean = np.mean(all_train_data, axis=0)
        global_std = np.std(all_train_data, axis=0)
        global_std = np.where(global_std < 1e-8, 1.0, global_std)
    else:
        all_data = np.vstack(sequences)
        global_mean = np.mean(all_data, axis=0)
        global_std = np.std(all_data, axis=0)
        global_std = np.where(global_std < 1e-8, 1.0, global_std)
    
    for seq in sequences:
        standardized_seq = (seq - global_mean) / global_std
        standardized_seqs.append(standardized_seq)
        
        scaler = {'mean': global_mean, 'std': global_std}
        scalers.append(scaler)
    
    return standardized_seqs, scalers

def inverse_standardize(standardized_seq: np.ndarray, scaler: dict, feature_idx: int = 1) -> np.ndarray:
    """反标准化特定特征"""
    return standardized_seq * scaler['std'][feature_idx] + scaler['mean'][feature_idx]

def load_and_preprocess_data(csv_files: List[str] = None) -> List[np.ndarray]:
    """数据加载和预处理"""
    if csv_files is None:
        csv_files = ['./src/p051/data_parse/cleaned_5_max_1.csv',
                     './src/p051/data_parse/cleaned_5_max_2.csv',
                     './src/p051/data_parse/cleaned_5_max_3.csv']
    
    sequences = []
    print("数据加载和趋势分析:")
    
    for file_path in csv_files:
        try:
            if not os.path.exists(file_path):
                print(f"警告: 文件 {file_path} 不存在")
                continue
                
            df = pd.read_csv(file_path, header=0)
            
            if df.shape[1] >= 2:
                seq_data = df.iloc[:, :2].values.astype(float)
                seq_data = seq_data[~np.any(np.isnan(seq_data), axis=1)]
                
                if len(seq_data) >= 120:
                    enhanced_data = calculate_trend_features(seq_data)
                    sequences.append(enhanced_data)
                    
                    values = seq_data[:, 1]
                    trend = np.polyfit(np.arange(len(values)), values, 1)[0]
                    print(f"文件 {file_path}: {len(seq_data)} 行数据, 趋势斜率: {trend:.6f}")
                else:
                    print(f"警告: 文件 {file_path} 数据长度不足120")
            else:
                print(f"警告: 文件 {file_path} 列数不足")
                    
        except Exception as e:
            print(f"错误: 读取文件 {file_path} 时发生错误: {e}")
    
    return sequences

def train_enhanced_model():
    """训练Informer模型"""
    sequences = load_and_preprocess_data()
    
    if len(sequences) < 2:
        print("错误: 数据不足")
        return None, [], [], sequences, []
    
    sequences_scaled, scalers = robust_standardize_sequences(sequences)
    
    train_sequences = sequences_scaled[:2]
    val_sequence = sequences_scaled[2] if len(sequences_scaled) > 2 else sequences_scaled[0]
    val_scaler = scalers[2] if len(scalers) > 2 else scalers[0]
    
    train_dataset = EnhancedSequenceDataset(
        train_sequences,
        input_length=120,
        output_length=150,  # 改为150 = label_len(50) + pred_len(100)，确保训练时y足够长
        stride=5,
        mode='train'
    )
    
    val_dataset = EnhancedSequenceDataset(
        [val_sequence],
        input_length=120,
        output_length=len(val_sequence) - 120,  # 验证时预测剩余全部
        mode='val'
    )
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    # 模型参数
    input_size = train_sequences[0].shape[1]  # 动态获取，如5
    seq_len = 120  # 输入长度
    label_len = 50  # 解码器起始长度（output/2）
    pred_len = 100  # 训练时预测长度
    d_model = 64  # 减小以适应小数据集
    n_heads = 4
    e_layers = 2
    d_layers = 1
    d_ff = 128
    dropout = 0.3
    factor = 3
    c_out = 1  # 只输出第二列
    
    model = Informer(
        enc_in=input_size,
        dec_in=1,  # 解码器输入只用值（起始0）
        c_out=c_out,
        seq_len=seq_len,
        label_len=label_len,
        pred_len=pred_len,
        factor=factor,
        d_model=d_model,
        n_heads=n_heads,
        e_layers=e_layers,
        d_layers=d_layers,
        d_ff=d_ff,
        dropout=dropout,
        activation='gelu',
        output_attention=False,
        distil=False  # 禁用distil，避免之前错误
    ).to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, factor=0.5)
    
    num_epochs = 150
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience = 20
    patience_counter = 0
    
    print("开始训练Informer模型...")
    print(f"训练样本数: {len(train_dataset)}")
    print(f"输入特征数: {input_size}")
    print(f"模型参数量: {sum(p.numel() for p in model.parameters())}")
    
    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)  # y_batch: [batch, 150, 1]
            
            optimizer.zero_grad()
            
            # Informer输入准备
            batch_size = x_batch.size(0)
            dec_inp = torch.zeros([batch_size, label_len + pred_len, 1]).float().to(device)  # 解码器起始0
            dec_inp[:, :label_len, :] = y_batch[:, :label_len, :]  # 填充label_len部分
            
            output = model(x_enc=x_batch, x_dec=dec_inp)
            loss = criterion(output, y_batch[:, label_len:, :])  # 只计算pred_len损失，现在y_batch[:, 50:, :] = [batch, 100, 1]
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_train_loss += loss.item()
        
        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val, y_val = x_val.to(device), y_val.to(device)  # y_val: [1, remaining_len, 1]
                
                # 动态调整pred_len为验证剩余长度
                val_pred_len = y_val.size(1)
                model.pred_len = val_pred_len
                dec_inp = torch.zeros([1, label_len + val_pred_len, 1]).float().to(device)
                # 验证时无label_len可用，所以全用0起始（或用输入的最后部分）
                dec_inp[:, :label_len, :] = x_val[:, -label_len:, 1:2]  # 用输入最后label_len的值填充
                
                output = model(x_enc=x_val, x_dec=dec_inp)
                val_loss = criterion(output, y_val)
                epoch_val_loss += val_loss.item()
        
        avg_val_loss = epoch_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        scheduler.step(avg_val_loss)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
        
        if epoch % 10 == 0 or epoch == num_epochs - 1:
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Epoch [{epoch}/{num_epochs}], Train Loss: {avg_train_loss:.6f}, '
                  f'Val Loss: {avg_val_loss:.6f}, LR: {current_lr:.6f}, '
                  f'Patience: {patience_counter}/{patience}')
        
        if patience_counter >= patience:
            print(f"早停触发于第 {epoch} 轮")
            break
    
    if os.path.exists('best_model.pth'):
        model.load_state_dict(torch.load('best_model.pth'))
    print(f"最佳验证损失: {best_val_loss:.6f}")
    
    return model, train_losses, val_losses, sequences, scalers

def evaluate_and_plot(model, sequences, scalers):
    """评估和绘图"""
    if len(sequences) < 3:
        print("数据不足")
        return
    
    model.eval()
    val_sequence = sequences[2]
    val_scaler = scalers[2]
    
    val_dataset = EnhancedSequenceDataset(
        [val_sequence],
        input_length=120,
        output_length=len(val_sequence) - 120,
        mode='val'
    )
    
    with torch.no_grad():
        for x_val, y_val in val_dataset:
            x_val = x_val.unsqueeze(0).to(device)
            y_val = y_val.squeeze(-1).numpy()  # [remaining_len]
            
            # 预测
            val_pred_len = len(y_val)
            model.pred_len = val_pred_len
            label_len = 50
            dec_inp = torch.zeros([1, label_len + val_pred_len, 1]).float().to(device)
            dec_inp[:, :label_len, :] = x_val[:, -label_len:, 1:2]  # 用输入最后部分填充
            
            predictions = model(x_enc=x_val, x_dec=dec_inp)
            predictions = predictions.squeeze(-1).cpu().numpy()[0]  # [pred_len]
            
            # 反标准化
            predictions_original = inverse_standardize(predictions, val_scaler)
            actual_original = inverse_standardize(y_val, val_scaler)
            
            break
    
    mse = np.mean((predictions_original - actual_original) ** 2)
    mae = np.mean(np.abs(predictions_original - actual_original))
    rmse = np.sqrt(mse)
    
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    time_actual = np.arange(len(actual_original))
    time_pred = np.arange(len(predictions_original)) + 120
    
    plt.plot(time_actual + 120, actual_original, label='实际值', linewidth=2, color='blue')
    plt.plot(time_pred, predictions_original, label='预测值', linewidth=2, color='red', linestyle='--')
    plt.axvline(x=120, color='gray', linestyle=':', alpha=0.7, label='预测起点')
    plt.xlabel('时间步')
    plt.ylabel('数值')
    plt.legend()
    plt.title('序列预测对比')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    errors = np.abs(predictions_original - actual_original)
    plt.plot(time_pred, errors, label='绝对误差', color='orange', linewidth=1)
    plt.xlabel('时间步')
    plt.ylabel('绝对误差')
    plt.legend()
    plt.title('预测误差')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 3)
    plt.plot(train_losses, label='训练损失', color='blue')
    plt.plot(val_losses, label='验证损失', color='red')
    plt.xlabel('训练轮次')
    plt.ylabel('损失')
    plt.legend()
    plt.title('训练历史')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 4)
    plt.scatter(actual_original, predictions_original, alpha=0.6, s=20)
    min_val = min(actual_original.min(), predictions_original.min())
    max_val = max(actual_original.max(), predictions_original.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
    plt.xlabel('实际值')
    plt.ylabel('预测值')
    plt.title('预测值 vs 实际值')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n=== 模型评估结果 ===")
    print(f"均方误差 (MSE): {mse:.6f}")
    print(f"均方根误差 (RMSE): {rmse:.6f}")
    print(f"平均绝对误差 (MAE): {mae:.6f}")
    print(f"预测长度: {len(predictions_original)} 个时间步")
    print(f"最终预测值: {predictions_original[-1]:.4f}")
    print(f"最终实际值: {actual_original[-1]:.4f}")
    print(f"最终误差: {abs(predictions_original[-1] - actual_original[-1]):.4f}")
    
    actual_trend = np.polyfit(np.arange(len(actual_original)), actual_original, 1)[0]
    pred_trend = np.polyfit(np.arange(len(predictions_original)), predictions_original, 1)[0]
    print(f"实际趋势斜率: {actual_trend:.6f}")
    print(f"预测趋势斜率: {pred_trend:.6f}")
    print(f"趋势捕捉误差: {abs(pred_trend - actual_trend):.6f}")

# 运行训练和评估
print("开始训练Informer模型...")
try:
    model, train_losses, val_losses, sequences, scalers = train_enhanced_model()
    
    if model is not None:
        evaluate_and_plot(model, sequences, scalers)
    else:
        print("训练失败")
        
except Exception as e:
    print(f"训练过程中出现错误: {e}")
    import traceback
    traceback.print_exc()