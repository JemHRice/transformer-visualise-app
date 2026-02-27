"""
Train a transformer with PyTorch on a language modeling task.
This creates a simple model that learns to predict the next word in a sequence.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
from torch.utils.data import DataLoader, Dataset


# Expanded sample text data for training
SAMPLE_TEXTS = [
    "the cat sat on the mat and watched birds",
    "the dog ran in the park and played ball",
    "a bird flew over the house and sang song",
    "the sun shines bright in the morning sky",
    "the moon appears in the night sky and glows",
    "the river flows to the sea and brings water",
    "the tree grows in the ground near house",
    "the book sits on the table in my room",
    "the fire burns very warm and bright fire",
    "the snow falls in winter and covers ground",
    "the rain comes from the clouds above us",
    "the wind blows through the trees very hard",
    "the star shines in the darkness at night",
    "the flower grows in spring with many colors",
    "the ice melts in summer when it is hot",
    "the sky turns blue when the sun comes out",
    "the cat sleeps on the warm mat at home",
    "the dog barks at the mailman every day",
    "the birds sing songs in the morning time",
    "the people walk in the park on sunny days",
    "the clouds float in the sky above us",
    "the grass grows green in the spring season",
] * 6  # Repeat for more training data


def build_vocab(texts):
    """Build vocabulary from texts."""
    words = set()
    for text in texts:
        words.update(text.split())
    word_to_idx = {word: idx for idx, word in enumerate(sorted(words))}
    idx_to_word = {idx: word for word, idx in word_to_idx.items()}
    return word_to_idx, idx_to_word


class LanguageDataset(Dataset):
    """Dataset for next-word prediction."""

    def __init__(self, texts, word_to_idx, seq_len=5):
        self.word_to_idx = word_to_idx
        self.seq_len = seq_len
        self.sequences = []
        self.targets = []

        for text in texts:
            words = text.split()
            for i in range(len(words) - seq_len):
                seq = [word_to_idx[w] for w in words[i : i + seq_len]]
                target = word_to_idx[words[i + seq_len]]
                self.sequences.append(seq)
                self.targets.append(target)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx]), torch.tensor(self.targets[idx])


class TransformerLM(nn.Module):
    """Simple transformer for language modeling."""

    def __init__(self, vocab_size, d_model=512, num_heads=4, seq_len=5):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = self._create_position_encoding(seq_len, d_model)

        # Transformer layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=2048,
            batch_first=True,
            dropout=0.1,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Output layer
        self.fc = nn.Linear(d_model, vocab_size)

    def _create_position_encoding(self, seq_len, d_model):
        """Create positional encoding."""
        position = torch.arange(seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        pe = torch.zeros(seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, x):
        # x shape: (batch, seq_len)
        x = self.embedding(x)  # (batch, seq_len, d_model)
        x = x + self.pos_encoding[:, : x.size(1), :].to(x.device)

        # Causal mask (prevent attend to future)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(x.size(1)).to(
            x.device
        )

        x = self.transformer(x, mask=causal_mask)

        # Use last token for prediction
        x = x[:, -1, :]  # (batch, d_model)
        x = self.fc(x)  # (batch, vocab_size)
        return x


def train_model(epochs=100, batch_size=8, learning_rate=0.001):
    """Train the transformer model."""

    # Prepare data
    word_to_idx, idx_to_word = build_vocab(SAMPLE_TEXTS)
    vocab_size = len(word_to_idx)

    print(f"Vocabulary size: {vocab_size}")
    print(f"Training on {len(SAMPLE_TEXTS)} text samples...")

    dataset = LanguageDataset(SAMPLE_TEXTS, word_to_idx, seq_len=5)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print(f"Created {len(dataset)} training sequences")

    # Create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerLM(vocab_size, d_model=512, num_heads=4, seq_len=5).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    print(f"\nTraining for {epochs} epochs on {device}...")
    for epoch in range(epochs):
        total_loss = 0

        for sequences, targets in dataloader:
            sequences = sequences.to(device)
            targets = targets.to(device)

            # Forward pass
            logits = model(sequences)
            loss = criterion(logits, targets)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

    print(f"\n✅ Training complete! Final loss: {avg_loss:.4f}")

    return model, word_to_idx, idx_to_word, device


def extract_weights_to_numpy(model):
    """Extract transformer layer weights and convert to NumPy format."""
    # Get the attention weights from the first transformer encoder layer
    transformer_layer = model.transformer.layers[0]

    # Extract multi-head attention weights
    # PyTorch stores them in a specific format, we'll extract the projection matrices
    mha = transformer_layer.self_attn

    d_model = model.d_model

    # Create weight matrices (PyTorch uses different format, so we'll create compatible ones)
    weights = {
        "W_q": mha.in_proj_weight[:d_model].cpu().detach().numpy(),
        "W_k": mha.in_proj_weight[d_model : 2 * d_model].cpu().detach().numpy(),
        "W_v": mha.in_proj_weight[2 * d_model :].cpu().detach().numpy(),
        "W_o": mha.out_proj.weight.cpu().detach().numpy(),
    }

    return weights


def save_weights(model, filename="trained_weights_pytorch.pkl"):
    """Save trained weights to file."""
    weights = extract_weights_to_numpy(model)

    with open(filename, "wb") as f:
        pickle.dump(weights, f)

    print(f"\nWeights saved to {filename}")


if __name__ == "__main__":
    # Train the model
    model, word_to_idx, idx_to_word, device = train_model(
        epochs=100, batch_size=8, learning_rate=0.001
    )

    # Save weights
    save_weights(model, "trained_weights.pkl")

    # Also save vocabulary for reference
    with open("vocab.pkl", "wb") as f:
        pickle.dump((word_to_idx, idx_to_word), f)

    print("✅ Training complete and weights saved!")
