"""
Purpose: Main training script for Siamese GNN models for smart contract clone detection.
This script supports both hierarchical and flat GNN architectures with contrastive loss training.
"""

import json
import os
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from tqdm import tqdm
import numpy as np
import random

from dataset import PairedGraphDataset, collate_fn_hierarchical
from model import HierarchicalGNN, FlatGNN

filelist = []
cfg_path = "GNNdata/noAbstact_proccessed_cfg"
for root, dirs, files in os.walk(cfg_path):
    for file in files:
        if file.endswith(".dot"):
            filelist.append(file.split("_")[0])

class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean(
            (label) * torch.pow(euclidean_distance, 2) +
            (1 - label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )
        return loss_contrastive

# TODO: Evaluate model performance on validation set
def evaluate(model, iterator, criterion, device):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for data1, data2, label in iterator:
            data1, data2, label = data1.to(device), data2.to(device), label.to(device).float()
            output1 = model(data1)
            output2 = model(data2)
            loss = criterion(output1, output2, label)
            epoch_loss += loss.item()
    return epoch_loss / len(iterator)

# TODO: Main training function for Siamese GNN models
def train(args):
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
        print("make dir: ", args.model_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dftrain = pd.read_csv(args.train_csv)
    dfval = pd.read_csv(args.val_csv)
    dftrain = dftrain[dftrain['contract_id'].isin(filelist) & dftrain['clone_contract_id'].isin(filelist)]
    dfval = dfval[dfval['contract_id'].isin(filelist) & dfval['clone_contract_id'].isin(filelist)]

    dftrain.to_csv(args.train_csv, index=False)
    dfval.to_csv(args.val_csv, index=False)

    train_dataset = PairedGraphDataset(dftrain, args.processed_dir)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,collate_fn = collate_fn_hierarchical)

    val_dataset = PairedGraphDataset(dfval, args.processed_dir)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,collate_fn=collate_fn_hierarchical)

    sample_data = train_dataset[0][0]
    node_feature_dim = sample_data.x.shape[1]
    output_dim = 128
    if args.model_type == 'hierarchical':
        model = HierarchicalGNN(
            node_feature_dim=node_feature_dim,
            gnn_hidden_dim=128,
            output_dim=output_dim
        ).to(device)
    elif args.model_type == 'base':
        model = FlatGNN(
            node_feature_dim=node_feature_dim,
            gnn_hidden_dim=128,
            output_dim=output_dim
        ).to(device)
    else:
        raise ValueError("未知的模型类型！请选择 'hierarchical' 或 'flat'。")
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = ContrastiveLoss(margin=args.margin)
    train_losses = []
    val_losses = []
    best_model = None
    best_val_loss = float('inf')
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for data1, data2, label in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            data1, data2, label = data1.to(device), data2.to(device), label.to(device).float()
            optimizer.zero_grad()

            output1 = model(data1)
            output2 = model(data2)
            loss = criterion(output1, output2, label)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        avg_val_loss = evaluate(model, val_loader, criterion, device)
        val_losses.append(avg_val_loss)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model = model
            torch.save(model.state_dict(), args.model_path + f'/best_model.pth')
        print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        log_data = {'train_loss': train_losses, 'val_loss': val_losses}
        with open('training_log.json', 'w') as f:
            json.dump(log_data, f, indent=4)

        torch.save(model.state_dict(), args.model_path + f'/{epoch}.pth')
        print(f"Model saved in {args.model_path}")

# TODO: Set random seeds for reproducible experiments
def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"所有随机种子已设置为: {seed_value}")

if __name__ == '__main__':
    set_seed(42)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='base', choices=['hierarchical', 'base'], help='选择要训练的GNN模型类型')
    parser.add_argument('--processed_dir', type=str, required=False, help='预处理好的图数据(.pt)文件夹',default='GNNdata/noAbstract_ptdata')
    parser.add_argument('--train_csv', type=str, required=False, help='训练集CSV文件',default='train.csv')
    parser.add_argument('--val_csv', type=str, help='验证集CSV文件',default='val.csv')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--margin', type=float, default=1.0, help='对比损失的margin值')
    parser.add_argument('--model_path', type=str, help='模型保存路径',default='noAbstractbasemodel')

    args = parser.parse_args()
    train(args)