import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class TextDataset(Dataset):
    """Dataset para textos tokenizados y etiquetas."""
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]


def make_collate_fn(pad_idx: int):
    """Genera una función collate con el padding_value especificado."""
    def collate_fn(batch):
        texts, labs = zip(*batch)
        padded = pad_sequence(texts, batch_first=True, padding_value=pad_idx)
        return padded.to(torch.long), torch.tensor(labs, dtype=torch.long)
    return collate_fn


class CNNTextClassifier(nn.Module):
    """Arquitectura CNN para clasificación de texto extraída de la celda 10 del notebook."""
    def __init__(
        self,
        vocab_size: int,
        emb_dim: int = 100,
        num_classes: int = 2,
        num_filters: int = 100,
        filter_sizes: list = [3, 4, 5],
        dropout: float = 0.5,
        pad_idx: int = 0
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (fs, emb_dim)) for fs in filter_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)

    def forward(self, x):
        emb = self.embedding(x).unsqueeze(1)
        conved = [torch.relu(conv(emb)).squeeze(3) for conv in self.convs]
        pooled = [torch.max(c, dim=2)[0] for c in conved]
        cat = torch.cat(pooled, dim=1)
        return self.fc(self.dropout(cat))
