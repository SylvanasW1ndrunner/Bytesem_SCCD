"""
Purpose: Defines paired graph dataset and hierarchical batch processing functions for Siamese GNN training.
Handles loading of preprocessed graph data and creates proper function mappings for hierarchical models.
"""

import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch, Data

class PairedGraphDataset(Dataset):
    def __init__(self, pairs_df: pd.DataFrame, processed_dir: str):
        self.processed_dir = processed_dir
        self.pairs_df = pairs_df.reset_index(drop=True)

    # TODO: Return the number of paired samples in the dataset
    def __len__(self):
        return len(self.pairs_df)

    # TODO: Load paired graph data for Siamese network training
    def __getitem__(self, idx):
        row = self.pairs_df.iloc[idx]
        id1, id2, label = row['contract_id'], row['clone_contract_id'], int(row['groundtruth'])

        data1_path = os.path.join(self.processed_dir, f"{id1}_cfg.pt")
        data2_path = os.path.join(self.processed_dir, f"{id2}_cfg.pt")

        data1 = torch.load(data1_path)
        data2 = torch.load(data2_path)

        return data1, data2, label

# TODO: Hierarchical batch processing function for paired graph data with function mappings
def collate_fn_hierarchical(batch):
    data1_list, data2_list, labels = [], [], []
    for _data1, _data2, _label in batch:
        data1_list.append(_data1)
        data2_list.append(_data2)
        labels.append(_label)

    # TODO: Process one side of paired data with hierarchical function mappings
    def process_side(data_list):
        batch = Batch.from_data_list(data_list)

        global_function_mapping = []
        function_offset = 0
        for data in data_list:
            valid_mapping = data.function_mapping[data.function_mapping >= 0]
            if len(valid_mapping) > 0:
                global_function_mapping.append(valid_mapping + function_offset)
                function_offset += data.num_functions

        if global_function_mapping:
            batch.global_function_mapping = torch.cat(global_function_mapping)
        else:
            batch.global_function_mapping = torch.empty(0, dtype=torch.long)

        func_to_graph_batch = []
        for i, data in enumerate(data_list):
            func_to_graph_batch.extend([i] * data.num_functions)

        batch.func_to_graph_batch = torch.tensor(func_to_graph_batch, dtype=torch.long)

        return batch

    batch1 = process_side(data1_list)
    batch2 = process_side(data2_list)

    labels = torch.tensor(labels, dtype=torch.float)

    return batch1, batch2, labels