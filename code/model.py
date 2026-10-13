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


class CascadedCNN1DBlock(nn.Module):
    """
    Paso 3: Módulo CNN-1D en Cascada para Extracción de Características locales.
    A la salida de cada rama de Transformer, conecta un bloque convolucional 1D:
      - 3 capas Conv1D con 64, 128 y 256 filtros y tamaños de kernel 3, 4 y 5 respectivamente (ReLU, padding='same').
      - Capas de Max-Pooling 1D intercaladas para reducir la dimensión reteniendo los valores máximos.
      - Capa de Average-Pooling 1D al final de cada rama para consolidar las características.
      - Reduce la representación de 768 dimensiones a un vector denso mejorado de 384 dimensiones.
    """
    def __init__(self, in_channels: int = 768, out_dim: int = 384, dropout: float = 0.2):
        super().__init__()
        # Capa 1: Conv1D (64 filtros, kernel 3, padding 'same') + ReLU + MaxPool1d
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=3, padding="same")
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        # Capa 2: Conv1D (128 filtros, kernel 4, padding 'same') + ReLU + MaxPool1d
        self.conv2 = nn.Conv1d(64, 128, kernel_size=4, padding="same")
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        # Capa 3: Conv1D (256 filtros, kernel 5, padding 'same') + ReLU
        self.conv3 = nn.Conv1d(128, 256, kernel_size=5, padding="same")
        self.relu3 = nn.ReLU()

        # Agrupamiento: Average-Pooling 1D al final para consolidar características
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        # Proyección final a 384 dimensiones
        self.fc = nn.Linear(256, out_dim)
        self.relu_fc = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, seq_len, 768] -> Conv1D espera [batch_size, channels, seq_len]
        x = x.transpose(1, 2)

        x = self.relu1(self.conv1(x))
        if x.size(2) > 1:
            x = self.pool1(x)

        x = self.relu2(self.conv2(x))
        if x.size(2) > 1:
            x = self.pool2(x)

        x = self.relu3(self.conv3(x))
        x = self.avgpool(x).squeeze(-1)  # [batch_size, 256]

        out = self.dropout(self.relu_fc(self.fc(x)))  # [batch_size, 384]
        return out


class HybridTransformerClassifier(nn.Module):
    """
    Modelo Híbrido Transformer + CNN-1D en Cascada para Clasificación en Ambiente Federado:
      - Rama 1 (MentalBERT): 'mental/mental-bert-base-uncased' (768 dimensiones).
      - Rama 2 (MelBERT): 'EhsanAghazadeh/melbert-roberta' (768 dimensiones).
      - Paso 3: Módulos CNN-1D en cascada para cada rama -> O_MentalBERT (384) y O_MelBERT (384).
      - Paso 4: Fusión O_fused = Concat(O_MentalBERT, O_MelBERT) en R^768.
      - Paso 5: Capa densa oculta (128 neuronas) + Capa de salida Softmax de 4 clases
               (Depresión, Ansiedad, BPD, PTSD).
    """
    def __init__(
        self,
        mental_model_name: str = "mental/mental-bert-base-uncased",
        melbert_model_name: str = "EhsanAghazadeh/melbert-roberta",
        num_classes: int = 4,
        freeze_transformers: bool = True,
        max_seq_len: int = 128,
        vocab_itos: list = None,
        pad_idx: int = 0,
        mental_model: nn.Module = None,
        melbert_model: nn.Module = None,
        mental_tokenizer = None,
        mel_tokenizer = None,
        hf_token: str = None
    ):
        super().__init__()
        self.num_classes = num_classes
        self.freeze_transformers = freeze_transformers
        self.max_seq_len = max_seq_len
        self.vocab_itos = vocab_itos
        self.pad_idx = pad_idx
        self.hf_token = hf_token

        # Cargar o reutilizar Rama 1: MentalBERT
        if mental_model is not None:
            self.mental_bert = mental_model
            self.mental_tokenizer = mental_tokenizer
        else:
            from transformers import AutoModel, AutoTokenizer
            self.mental_tokenizer = AutoTokenizer.from_pretrained(mental_model_name, token=hf_token)
            self.mental_bert = AutoModel.from_pretrained(mental_model_name, token=hf_token)

        # Cargar o reutilizar Rama 2: MelBERT
        if melbert_model is not None:
            self.melbert = melbert_model
            self.mel_tokenizer = mel_tokenizer
        else:
            from transformers import AutoModel, AutoTokenizer
            self.mel_tokenizer = AutoTokenizer.from_pretrained(melbert_model_name, token=hf_token)
            self.melbert = AutoModel.from_pretrained(melbert_model_name, token=hf_token)

        # Si se congelan los transformers, no computan gradientes en clientes federados
        if self.freeze_transformers:
            for p in self.mental_bert.parameters():
                p.requires_grad = False
            for p in self.melbert.parameters():
                p.requires_grad = False

        # Paso 3: Módulos CNN-1D en cascada para cada rama (768 -> 384)
        self.cnn_mental = CascadedCNN1DBlock(in_channels=768, out_dim=384)
        self.cnn_mel = CascadedCNN1DBlock(in_channels=768, out_dim=384)

        # Paso 5: Capa de clasificación densamente conectada
        # Procesa vector fusionado de 768 dimensiones a través de capa oculta de 128 neuronas
        self.fc_hidden = nn.Linear(768, 128)
        self.relu_hidden = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

        # Capa de salida: 4 clases (Depresión, Ansiedad, BPD, PTSD)
        self.classifier = nn.Linear(128, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def _prepare_inputs(self, x):
        """Convierte entradas (textos o tensores de IDs) en inputs listos para MentalBERT y MelBERT."""
        device = next(self.parameters()).device

        # Caso 1: Lista de strings de texto
        if isinstance(x, (list, tuple)) and len(x) > 0 and isinstance(x[0], str):
            texts = list(x)
        # Caso 2: Tensor de IDs numéricos -> Si hay vocab_itos, decodificar a texto
        elif isinstance(x, torch.Tensor):
            if self.vocab_itos is not None:
                texts = []
                for seq in x:
                    words = [
                        self.vocab_itos[tok.item()]
                        for tok in seq
                        if tok.item() < len(self.vocab_itos)
                        and self.vocab_itos[tok.item()] not in ("<pad>", "<unk>")
                    ]
                    texts.append(" ".join(words) if words else "<pad>")
            else:
                # Si no hay vocab_itos, pasar directamente como input_ids
                input_ids = x.to(device)
                attn_mask = (input_ids != self.pad_idx).long().to(device)
                return {"input_ids": input_ids, "attention_mask": attn_mask}, {"input_ids": input_ids, "attention_mask": attn_mask}
        else:
            texts = [str(t) for t in x]

        # Tokenización adaptada para cada transformer
        mental_inputs = self.mental_tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_seq_len,
            return_tensors="pt"
        )
        mel_inputs = self.mel_tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_seq_len,
            return_tensors="pt"
        )
        mental_inputs = {k: v.to(device) for k, v in mental_inputs.items()}
        mel_inputs = {k: v.to(device) for k, v in mel_inputs.items()}
        return mental_inputs, mel_inputs

    def forward(self, x) -> torch.Tensor:
        mental_inputs, mel_inputs = self._prepare_inputs(x)

        # Paso 1 y 2: Generar embeddings contextuales de 768 dimensiones
        with torch.set_grad_enabled(not self.freeze_transformers):
            emb_mental = self.mental_bert(**mental_inputs).last_hidden_state  # [B, L, 768]
            emb_mel = self.melbert(**mel_inputs).last_hidden_state           # [B, L, 768]

        # Paso 3: Módulos CNN-1D en cascada (768 -> 384 por rama)
        o_mental = self.cnn_mental(emb_mental)  # [B, 384]
        o_mel = self.cnn_mel(emb_mel)            # [B, 384]

        # Paso 4: Fusión y concatenación: O_fused = Concat(O_MentalBERT, O_MelBERT) in R^768
        o_fused = torch.cat([o_mental, o_mel], dim=1)  # [B, 768]

        # Paso 5: Capa de clasificación densamente conectada (768 -> 128 -> num_classes)
        h = self.dropout(self.relu_hidden(self.fc_hidden(o_fused)))
        logits = self.classifier(h)
        return logits

    def predict_proba(self, x) -> torch.Tensor:
        """Calcula probabilidades con Softmax sobre las 4 clases."""
        logits = self.forward(x)
        return self.softmax(logits)
