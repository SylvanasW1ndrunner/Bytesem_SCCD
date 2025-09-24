"""
Purpose: Prepares training data for sequence-to-sequence autoencoder.
Handles vocabulary building, tokenization, and data loading for instruction sequence encoding.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
from tqdm import tqdm
import pickle
import os

class Vocabulary:
    def __init__(self, min_freq=2):
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.min_freq = min_freq

    # TODO: Return vocabulary size
    def __len__(self):
        return len(self.itos)

    # TODO: Build vocabulary from sentence list with frequency filtering
    def build_vocabulary(self, sentence_list):
        print("词汇表不存在，正在从头构建...")
        word_counts = Counter(word for sentence in tqdm(sentence_list, desc="Counting tokens") for word in sentence)

        idx = 4
        for word, count in word_counts.items():
            if count >= self.min_freq:
                self.stoi[word] = idx
                self.itos[idx] = word
                idx += 1
        print(f"词汇表构建完成，总词数: {len(self.itos)}")

    # TODO: Convert token sequence to numerical IDs
    def numericalize(self, text_tokens: list) -> list:
        return [self.stoi.get(token, self.stoi["<UNK>"]) for token in text_tokens]

    # TODO: Save vocabulary to pickle file
    def save(self, path="tokenizer.pkl"):
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print(f"词汇表已保存到: {path}")

    # TODO: Load vocabulary from pickle file
    @classmethod
    def load(cls, path="tokenizer.pkl"):
        with open(path, 'rb') as f:
            vocab = pickle.load(f)
        print(f"从 '{path}' 加载词汇表成功。")
        return vocab

class OpcodeDataset(Dataset):
    def __init__(self, corpus_path, vocab: Vocabulary):
        self.vocab = vocab
        with open(corpus_path, 'r', encoding='utf-8') as f:
            self.lines = [line.strip().split() for line in f.readlines() if line.strip()]

    # TODO: Return dataset size
    def __len__(self):
        return len(self.lines)

    # TODO: Get tokenized sequence with SOS/EOS tokens
    def __getitem__(self, index):
        tokens = self.lines[index]
        numericalized_tokens = self.vocab.numericalize(tokens)
        return torch.tensor([self.vocab.stoi["<SOS>"]] + numericalized_tokens + [self.vocab.stoi["<EOS>"]])

class PadCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    # TODO: Collate function for padding variable-length sequences
    def __call__(self, batch):
        source_seqs = [item for item in batch]
        padded_sources = pad_sequence(source_seqs, batch_first=False, padding_value=self.pad_idx)
        source_lens = torch.tensor([len(seq) for seq in source_seqs], dtype=torch.int64)
        return padded_sources, source_lens

# TODO: Get dataloader and vocabulary, create if not exists
def get_dataloader(corpus_path: str, tokenizer_path: str, batch_size: int = 32, min_freq: int = 2) -> (DataLoader, Vocabulary):
    if os.path.exists(tokenizer_path):
        vocab = Vocabulary.load(tokenizer_path)
    else:
        print(f"词汇表文件 '{tokenizer_path}' 未找到。")
        with open(corpus_path, 'r', encoding='utf-8') as f:
            sentences = [line.strip().split() for line in f.readlines() if line.strip()]
        vocab = Vocabulary(min_freq)
        vocab.build_vocabulary(sentences)
        vocab.save(tokenizer_path)

    dataset = OpcodeDataset(corpus_path, vocab)
    pad_idx = vocab.stoi["<PAD>"]

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=PadCollate(pad_idx=pad_idx)
    )

    return dataloader, vocab

if __name__ == '__main__':
    CORPUS_FILE = r'C:\Users\zkjg\Downloads\13756064\clone_detection_replication_package\GNNMethod\no_abstract_corpus_deduplicated.txt'
    TOKENIZER_FILE = 'noAbstract_tokenizer.pkl'

    if not os.path.exists(CORPUS_FILE):
        print(f"错误: 语料库文件 '{CORPUS_FILE}' 不存在，请先准备好数据。")
    else:
        dataloader, vocab = get_dataloader(CORPUS_FILE, TOKENIZER_FILE, batch_size=4)
        print(f"\n成功获取DataLoader。词汇表大小: {len(vocab)}")
        src_batch, src_len_batch = next(iter(dataloader))
        print(f"\n一个batch的源数据形状 (Seq_Len, Batch_Size): {src_batch.shape}")
        print(f"一个batch的源数据长度 (Batch_Size): {src_len_batch}")
        print("\n第一个样本 (数字形式):")
        print(src_batch[:, 0])
        print("\n第一个样本 (Token形式):")
        print([vocab.itos[idx.item()] for idx in src_batch[:, 0]])