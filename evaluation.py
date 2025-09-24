"""
Purpose: Comprehensive evaluation script for trained Siamese GNN models with detailed performance reports 
and publication-quality visualizations. Supports both standard evaluation and robustness testing modes.
"""

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import argparse
import os
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_fscore_support
from tqdm import tqdm
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from dataset import PairedGraphDataset, collate_fn_hierarchical
from model import HierarchicalGNN, FlatGNN
from train import ContrastiveLoss

png_path = ""

filelist = []
cfg_path = "GNNdata/proccessed_cfg"
for root, dirs, files in os.walk(cfg_path):
    for file in files:
        if file.endswith(".dot"):
            filelist.append(file.split("_")[0])

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 28,
    'axes.labelsize': 28,
    'axes.titlesize': 28,
    'xtick.labelsize': 26,
    'ytick.labelsize': 26,
    'legend.fontsize': 16,
    'legend.title_fontsize': 18,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

# TODO: Plot publication-quality distance distribution by clone type using violin and strip plots
def plot_distance_by_type(df_results: pd.DataFrame, output_dir: str):
    print("正在生成出版级质量的、按克隆类型的距离分布图...")
    
    df_to_plot = df_results.copy()
    label_map = {
        0: 'Non-Clone', 1: 'Type-1', 2: 'Type-2',
        3: 'Type-3', 4: 'Type-4'
    }
    category_order = ['Type-4', 'Type-3', 'Type-2', 'Type-1', 'Non-Clone']
    df_to_plot['Category'] = pd.Categorical(df_to_plot['type'].map(label_map),
                                            categories=category_order,
                                            ordered=True)

    plt.figure(figsize=(14, 8))
    ax = plt.gca()
    sns.violinplot(
        data=df_to_plot,
        x='distance',
        y='Category',
        ax=ax,
        palette='viridis_r',
        orient='h',
        inner=None,
        linewidth=0,
        alpha=0.4
    )

    sns.stripplot(
        data=df_to_plot,
        x='distance',
        y='Category',
        ax=ax,
        jitter=0.25,
        size=6,
        alpha=0.6,
        color=".3"
    )

    sns.despine(ax=ax, left=True)
    ax.grid(axis='x', linestyle='--', linewidth=2.0, color='black', alpha=0.6)
    ax.set_xlabel('')
    ax.set_ylabel('')
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "distance_by_type_distribution.svg")
    plt.savefig(output_path, format='svg')
    print(f"按类型距离分布图已保存为高质量SVG文件: {output_path}")
    plt.close()

# TODO: Plot training and validation loss curves from training log
def plot_loss_curve(log_path: str):
    with open(log_path, 'r') as f:
        log_data = json.load(f)

    plt.figure(figsize=(10, 6))
    plt.plot(log_data['train_loss'], label='Training Loss', marker='o', linestyle='-')
    if log_data.get('val_loss'):
        plt.plot(log_data['val_loss'], label='Validation Loss', marker='x', linestyle='--')
    plt.title('Training & Validation Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Contrastive Loss')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.savefig(png_path + "/loss_curve_publication.png")
    print("训练/验证损失曲线图已保存到 loss_curve_publication.png")

# TODO: Generate confusion matrix visualization and save detailed metrics
def plot_confusion_matrix(y_true, y_pred, class_names=["Not Clone", "Clone"]):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.savefig(png_path+"/confusion_matrix_publication.png")
    print("混淆矩阵图已保存到 confusion_matrix_publication.png")
    
    print("\n--- 混淆矩阵详细数据 ---")
    print(f"  True Positives (TP): {tp}")
    print(f"  True Negatives (TN): {tn}")
    print(f"  False Positives (FP): {fp} (Type I Error)")
    print(f"  False Negatives (FN): {fn} (Type II Error)")
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    with open (png_path+"/confusion_matrix.txt", 'w') as f:
        f.write(f"  True Positives (TP): {tp}\n")
        f.write(f"  True Negatives (TN): {tn}\n")
        f.write(f"  False Positives (FP): {fp} (Type I Error)\n")
        f.write(f"  False Negatives (FN): {fn} (Type II Error)\n")
        f.write(f"  Precision: {precision:.4f}\n")
        f.write(f"  Recall: {recall:.4f}\n")
        f.write(f"  F1 Score: {f1:.4f}\n")

# TODO: Plot distance distribution histogram for clone vs non-clone pairs
def plot_distance_distribution(y_true, y_pred_dist):
    df = pd.DataFrame({'distance': y_pred_dist, 'is_clone': y_true})

    plt.figure(figsize=(12, 7))
    ax = sns.histplot(data=df, x='distance', hue='is_clone', kde=True, bins=50, palette='viridis')
    ax.grid(False)
    plt.xlabel('Euclidean Distance in Embedding Space')
    plt.ylabel('Frequency')
    plt.legend(title='Pair Type', labels=['Clone (Positive)', 'Not Clone (Negative)'])
    plt.savefig(png_path + "/distance_distribution.svg",format='svg')
    print("样本对距离分布图已保存到 distance_distribution.svg")

# TODO: Plot performance metrics (precision, recall, F1) for each clone type
def plot_per_type_metrics(test_df, y_pred_class):
    clone_df = test_df[test_df['groundtruth'] == 1].copy()
    clone_df['predicted_class'] = y_pred_class[clone_df.index]

    report = {}
    for clone_type in sorted(clone_df['type'].unique()):
        type_df = clone_df[clone_df['type'] == clone_type]
        if len(type_df) == 0: continue

        p, r, f1, _ = precision_recall_fscore_support(
            type_df['groundtruth'],
            type_df['predicted_class'],
            average='binary',
            pos_label=1,
            zero_division=0
        )
        report[f'Type-{int(clone_type)}'] = {'Precision': p, 'Recall': r, 'F1-Score': f1}

    report_df = pd.DataFrame.from_dict(report, orient='index')

    report_df.plot(kind='bar', figsize=(12, 7), rot=0)
    plt.title('Performance Metrics per Clone Type')
    plt.xlabel('Clone Type')
    plt.ylabel('Score')
    plt.ylim(0, 1.05)
    plt.grid(axis='y', linestyle='--')
    plt.legend(loc='lower right')
    plt.savefig(png_path + "/per_type_metrics.png")
    print("各类型克隆性能条形图已保存到 per_type_metrics.png")
    print("\n--- 各类型克隆的独立召回率 (Recall) ---")
    print(report_df[['Recall']])

# TODO: Plot dataset distribution pie chart showing clone type proportions
def plot_dataset_distribution(csv_files: dict):
    all_data = []
    for name, path in csv_files.items():
        print(path)
        df = pd.read_csv(path)
        df['split'] = name
        all_data.append(df)

    full_df = pd.concat(all_data)
    type_counts = full_df['type'].value_counts()

    plt.figure(figsize=(8, 8))
    plt.pie(type_counts, labels=type_counts.index, autopct='%1.1f%%', startangle=140)
    plt.title('Distribution of Clone Types in the Entire Dataset')
    plt.ylabel('')
    plt.savefig(png_path + "/dataset_distribution.png")
    print("数据集分布图已保存到 dataset_distribution.png")

# TODO: Plot ROC curve and calculate AUC for performance evaluation
def plot_roc_curve(y_true, y_pred_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:0.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig(png_path + "/roc_curve.png")
    print(f"ROC曲线图已保存到 roc_curve.png (AUC = {roc_auc:.3f})")
    return thresholds, tpr, fpr

# TODO: Main evaluation function for trained Siamese GNN models
def evaluate(args):
    global png_path
    png_path = args.output_dir
    if not os.path.exists(png_path):
        os.mkdir(png_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    df_test = pd.read_csv(args.test_csv)
    df_test = df_test[df_test['contract_id'].isin(filelist) & df_test['clone_contract_id'].isin(filelist)]

    df_test.to_csv(args.test_csv, index=False)
    csv_files = {'train': args.train_csv, 'val': args.val_csv, 'test': args.test_csv}
    plot_dataset_distribution(csv_files)
    if os.path.exists(args.log_path):
        plot_loss_curve(args.log_path)

    test_dataset = PairedGraphDataset(df_test, args.processed_dir)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn_hierarchical)
    if args.robust == True:
        noise_test_loader = test_loader
        origin_test_dataset = PairedGraphDataset(df_test, "GNNdata/ptdata")
        origin_test_loader = DataLoader(origin_test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn_hierarchical)
    
    sample_data = test_dataset[0][0]
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
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    print(f"模型 '{args.model_path}' 加载成功。")

    y_true = []
    y_pred_distances = []
    all_embeddings = []
    all_type = []
    all_type = df_test['type'].tolist()
    print("在测试集上进行预测...")
    if args.robust == False:
        with torch.no_grad():
            for data1, data2, label in tqdm(test_loader, desc="Evaluating"):
                data1, data2, label = data1.to(device), data2.to(device), label.to(device).float()

                output1 = model(data1)
                output2 = model(data2)
                distance = F.pairwise_distance(output1, output2)
                y_pred_distances.extend(distance.cpu().numpy())
                y_true.extend(label.cpu().numpy())
        print("y_pred_distances:", len(y_pred_distances), "y_true:",( len(y_true)))
        y_true = np.array(y_true)
        y_pred_scores = 1 - np.array(y_pred_distances)

        thresholds, tpr, fpr = plot_roc_curve(y_true, y_pred_scores)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        print(f"\n根据ROC曲线找到的最佳分类阈值为: {optimal_threshold:.4f}")

        y_pred_class = (y_pred_scores >= optimal_threshold).astype(int)
        df_test['y_pred_class'] = y_pred_class
        print("\n--- 总体性能指标 ---")
        print(classification_report(y_true, y_pred_class, target_names=["Not Clone", "Clone"]))
        with open(png_path + "/performance.txt", "w") as f:
            f.write(classification_report(y_true, y_pred_class, target_names=["Not Clone", "Clone"]))
        results_df = df_test.iloc[:, -3:]
        plot_distance_distribution(y_true, y_pred_distances)
        plot_confusion_matrix(y_true, y_pred_class)

        print("\n--- 各类型克隆的独立召回率 (Recall) ---")
        test_df = pd.read_csv(args.test_csv)
        test_df['predicted_class'] = y_pred_class
        clone_df = test_df[test_df['groundtruth'] == 1].copy()
        clone_df['predicted_class'] = y_pred_class[clone_df.index]
        with open(png_path + "/performance.txt", "a") as f:
            f.write("\n--- 各类型克隆的独立召回率 (Recall) ---\n")
            for clone_type in sorted(clone_df['type'].unique()):
                type_df = clone_df[clone_df['type'] == clone_type]
                true_positives = type_df[type_df['predicted_class'] == 1].shape[0]
                total_positives = type_df.shape[0]
                recall = true_positives / total_positives if total_positives > 0 else 0
                print(f"  Type-{int(clone_type)} Recall: {recall:.4f} ({true_positives} / {total_positives})")
                f.write(f"  Type-{int(clone_type)} Recall: {recall:.4f} ({true_positives} / {total_positives})\n")
        df_test['distance'] = y_pred_distances
        plot_distance_by_type(df_test, png_path)
        test_df = pd.read_csv(args.test_csv)
        plot_per_type_metrics(test_df, y_pred_class)
    else:
        all_true_labels = []
        all_dist_orig = []
        all_dist_noisy = []
        all_robustness_scores = []

        print("\n--- 开始鲁棒性测试模式 ---")

        with torch.no_grad():
            for (noise_data1, noise_data2, label1), (origin_data1, origin_data2, label2) in tqdm(zip(noise_test_loader, origin_test_loader), desc="Evaluating Robustness", total=len(test_loader)):

                noise_data1, noise_data2, label = noise_data1.to(device), noise_data2.to(device), label1.to(device).float()
                origin_data1, origin_data2 = origin_data1.to(device), origin_data2.to(device)

                origin_output1 = model(origin_data1)
                origin_output2 = model(origin_data2)
                noise_output1 = model(noise_data1)
                noise_output2 = model(noise_data2)

                dist_orig = F.pairwise_distance(origin_output1, origin_output2)
                dist_noisy = F.pairwise_distance(noise_output1, noise_output2)

                sim_1 = F.cosine_similarity(origin_output1, noise_output1)
                sim_2 = F.cosine_similarity(origin_output2, noise_output2)
                avg_robustness_batch = (sim_1 + sim_2) / 2

                all_true_labels.extend(label.cpu().numpy())
                all_dist_orig.extend(dist_orig.cpu().numpy())
                all_dist_noisy.extend(dist_noisy.cpu().numpy())
                all_robustness_scores.extend(avg_robustness_batch.cpu().numpy())

        results_df = pd.DataFrame({
            'groundtruth': all_true_labels,
            'dist_orig': all_dist_orig,
            'dist_noisy': all_dist_noisy,
            'robustness_score': all_robustness_scores
        })

        results_df.to_csv(args.output_dir + "/robustness_detailed_results.csv", index=False)
        print("\n详细的鲁棒性测试原始数据已保存到: robustness_detailed_results.csv")

        with open("robustness_report.txt", "w", encoding="utf-8") as report_file:

            report_file.write("="*25 + " 任务一：扰动下的克隆检测性能 " + "="*25 + "\n\n")

            scores_noisy = 1 - (results_df['dist_noisy'] / (results_df['dist_noisy'].max() + 1e-9))
            y_true = results_df['groundtruth']

            fpr, tpr, thresholds = roc_curve(y_true, scores_noisy)
            optimal_threshold = thresholds[np.argmax(tpr - fpr)]
            y_pred_class = (scores_noisy >= optimal_threshold).astype(int)

            print(f"\n--- [任务一] 扰动后数据的分类性能报告 (阈值={optimal_threshold:.4f}) ---")
            report_str = classification_report(y_true, y_pred_class, target_names=["Not Clone", "Clone"])
            print(report_str)
            report_file.write(f"最佳分类阈值: {optimal_threshold:.4f}\n")
            report_file.write(report_str + "\n")

            cm = confusion_matrix(y_true, y_pred_class)
            tn, fp, fn, tp = cm.ravel()
            report_file.write("\n--- 混淆矩阵详细数据 ---\n")
            report_file.write(f"  True Positives (TP): {tp}\n")
            report_file.write(f"  True Negatives (TN): {tn}\n")
            report_file.write(f"  False Positives (FP): {fp}\n")
            report_file.write(f"  False Negatives (FN): {fn}\n")

            plot_confusion_matrix(y_true, y_pred_class)

            report_file.write("\n" + "="*25 + " 任务二：嵌入向量稳定性分析 " + "="*25 + "\n\n")

            robustness_stats = results_df['robustness_score'].describe()
            print("\n--- [任务二] 鲁棒性得分 (余弦相似度) 统计 ---")
            print(robustness_stats)
            report_file.write("鲁棒性得分 (越高越好):\n")
            report_file.write(robustness_stats.to_string())

            plot_robustness_score_distribution(results_df)
            plot_distance_shift_scatter(results_df)

        print(f"\n完整的鲁棒性评估报告已保存到: robustness_report.txt")
        print("相关的图表已生成并保存为PNG文件。")

# TODO: Plot robustness score distribution histogram for embedding stability analysis
def plot_robustness_score_distribution(df):
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='robustness_score', kde=True, bins=50, color='dodgerblue')
    score_mean = df['robustness_score'].mean()
    plt.axvline(score_mean, color='r', linestyle='--', label=f'Mean: {score_mean:.3f}')
    plt.title('Robustness Score Distribution (Original vs. Noisy)')
    plt.xlabel('Cosine Similarity (Higher is Better)')
    plt.ylabel('Frequency')
    plt.xlim(df['robustness_score'].min() * 0.95, 1.0)
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.savefig(args.output_dir + "/robustness_score_distribution.png")
    plt.close()

# TODO: Plot scatter plot showing distance changes before and after perturbation
def plot_distance_shift_scatter(df):
    plt.figure(figsize=(8, 8))
    plot = sns.scatterplot(data=df, x='dist_orig', y='dist_noisy', hue='groundtruth', palette='coolwarm', alpha=0.7)
    min_val = min(df['dist_orig'].min(), df['dist_noisy'].min())
    max_val = max(df['dist_orig'].max(), df['dist_noisy'].max())
    plot.plot([min_val, max_val], [min_val, max_val], ls="--", c=".3", label="y=x (No Change)")
    plt.title('Distance of Pairs Before vs. After Perturbation')
    plt.xlabel('Distance of Original Pairs')
    plt.ylabel('Distance of Noisy Pairs')
    handles, labels = plot.get_legend_handles_labels()
    plot.legend(handles=handles, title='Ground Truth', labels=['Not Clone (0)', 'Clone (1)'])
    plt.grid(True)
    plt.savefig(args.output_dir + "/robustness_distance_shift.png")
    plt.close()

# TODO: Plot similarity score distribution scatter plot for threshold analysis
def plot_similarity_scatter(df_results, optimal_threshold, output_dir):
    plt.figure(figsize=(12, 6))
    sns.stripplot(data=df_results, x='similarity_score', y='groundtruth_str', jitter=0.2, alpha=0.6, palette='viridis')
    plt.axvline(optimal_threshold, color='r', linestyle='--', label=f'Optimal Threshold = {optimal_threshold:.3f}')
    plt.title('Similarity Score Distribution for Clone vs. Non-Clone Pairs')
    plt.xlabel('Similarity Score (Higher is more likely a clone)')
    plt.ylabel('Ground Truth')
    plt.legend()
    plt.grid(axis='x')
    plt.savefig(os.path.join(output_dir, "similarity_scatter_plot.png"))
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='评估孪生GNN模型并生成图表')
    parser.add_argument('--model_type', type=str, default='hierarchical', choices=['hierarchical', 'base'], help='选择要训练的GNN模型类型')
    parser.add_argument('--model_path', type=str, required=False, help='训练好的模型文件路径 (.pt)',default='noAbstractmodel/best_model.pth')
    parser.add_argument('--processed_dir', type=str, required=False, help='预处理好的图数据(.pt)文件夹',default='GNNdata/noABstract_ptdata')
    parser.add_argument('--train_csv', type=str, required=False, help='训练集CSV文件路径', default='train.csv')
    parser.add_argument('--val_csv', type=str, required=False, help='验证集CSV文件路径',default='val.csv')
    parser.add_argument('--test_csv', type=str, required=False, help='测试集CSV文件路径',default='test.csv')
    parser.add_argument('--log_path', type=str, default='training_log.json', help='训练日志文件路径')
    parser.add_argument('--batch_size', type=int, default=128, help='评估时使用的批大小')
    parser.add_argument('--robust', type=bool, default=False, help='评估时使用的线程数')
    parser.add_argument('--output_dir', type=str,default='noAbstracthierReport' ,help='输出文件夹')
    args = parser.parse_args()
    png_path = args.output_dir
    evaluate(args)