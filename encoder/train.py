"""
Purpose: Trains sequence-to-sequence autoencoder for encoding smart contract instruction sequences.
Creates compressed representations of bytecode instructions for use in GNN node features.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import time
from prepare_data import get_dataloader, Vocabulary
from model import Encoder, Decoder, Seq2SeqAutoencoder

CORPUS_FILE = '../no_abstract_corpus_deduplicated.txt'
TOKENIZER_FILE = 'noAbstract_tokenizer.pkl'
BATCH_SIZE = 64
EMB_DIM = 128
HID_DIM = 256
N_LAYERS = 2
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5
N_EPOCHS = 10
CLIP = 1
model_path = 'noAbstractModel'

# TODO: Train the sequence-to-sequence autoencoder model for one epoch
def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    for i, (src, src_len) in enumerate(tqdm(iterator, desc="Training")):
        src = src.to(device)
        trg = src

        optimizer.zero_grad()

        output = model(src, src_len, trg)

        output_dim = output.shape[-1]

        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)

        loss = criterion(output, trg)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_dataloader, vocab = get_dataloader(CORPUS_FILE, TOKENIZER_FILE, BATCH_SIZE)
    INPUT_DIM = len(vocab)
    OUTPUT_DIM = len(vocab)
    PAD_IDX = vocab.stoi['<PAD>']

    enc = Encoder(INPUT_DIM, EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
    dec = Decoder(OUTPUT_DIM, EMB_DIM, HID_DIM * 2, N_LAYERS, DEC_DROPOUT)
    model = Seq2SeqAutoencoder(enc, dec, device).to(device)

    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    for epoch in range(N_EPOCHS):
        start_time = time.time()

        train_loss = train(model, train_dataloader, optimizer, criterion, CLIP)

        end_time = time.time()
        epoch_mins, epoch_secs = divmod(end_time - start_time, 60)

        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs:.0f}s')
        print(f'\tTrain Loss: {train_loss:.3f} | PPL: {torch.exp(torch.tensor(train_loss)):7.3f}')

        torch.save(model.state_dict(), model_path + f'/autoencoder_epoch_{epoch+1}.pt')
        print(f"模型已保存: autoencoder_epoch_{epoch+1}.pt")