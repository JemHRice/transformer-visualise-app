"""
Train a simple transformer on next-word prediction task.
This script trains the model and saves the weights for use in the app.
"""

import numpy as np
import pickle
from transformer import (
    softmax,
    scaled_dot_product_attention,
    positional_encoding,
)


def generate_training_data(num_samples=100, vocab_size=50, seq_len=8):
    """
    Generate simple training data for next-word prediction.

    Returns:
        X: (num_samples, seq_len, vocab_size) - one-hot encoded sequences
        y: (num_samples,) - next word predictions
    """
    X = np.random.randint(0, vocab_size, size=(num_samples, seq_len))
    y = np.random.randint(0, vocab_size, size=num_samples)
    return X, y


def embeddings_from_onehot(X, embedding_dim):
    """Convert integer sequences to random embeddings."""
    num_samples, seq_len = X.shape
    embeddings = np.random.randn(np.max(X) + 1, embedding_dim) * 0.01
    X_embedded = np.array([embeddings[x] for x in X])
    return X_embedded, embeddings


def train_transformer(
    d_model=512,
    num_heads=4,
    num_epochs=50,
    learning_rate=0.001,
    num_samples=200,
):
    """
    Train a transformer on next-word prediction.

    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        num_epochs: Training epochs
        learning_rate: Learning rate
        num_samples: Number of training samples

    Returns:
        trained_weights: Dict of trained weight matrices
    """
    print(f"Starting training for {num_epochs} epochs...")

    # Generate data
    X, y = generate_training_data(num_samples=num_samples, seq_len=8)
    seq_len = X.shape[1]

    # Initialize weights
    d_k = d_model // num_heads

    W_q = np.random.randn(d_model, d_model) * 0.01
    W_k = np.random.randn(d_model, d_model) * 0.01
    W_v = np.random.randn(d_model, d_model) * 0.01
    W_o = np.random.randn(d_model, d_model) * 0.01
    W_final = np.random.randn(d_model, 50) * 0.01  # Output layer

    # Positional encoding
    pe = positional_encoding(seq_len, d_model)

    # Training loop
    for epoch in range(num_epochs):
        epoch_loss = 0

        for i in range(num_samples):
            # Prepare input
            x_seq = X[i]  # Shape: (seq_len,)
            embeddings = np.random.randn(np.max(x_seq) + 1, d_model) * 0.01
            X_embedded = embeddings[x_seq]  # Shape: (seq_len, d_model)

            # Add positional encoding
            X_with_pos = X_embedded + pe[:seq_len]

            # Forward pass (simplified - single head for speed)
            Q = X_with_pos @ W_q
            K = X_with_pos @ W_k
            V = X_with_pos @ W_v

            # Use last token's attention
            Q_last = Q[-1:] / np.sqrt(d_k)  # (1, d_model)
            K_all = K / np.sqrt(d_k)  # (seq_len, d_model)
            V_all = V  # (seq_len, d_model)

            scores = Q_last @ K_all.T  # (1, seq_len)
            attn_weights = softmax(scores)  # (1, seq_len)
            context = attn_weights @ V_all  # (1, d_model)

            output = context @ W_o  # (1, d_model)
            logits = output @ W_final  # (1, 50)

            # Compute loss
            pred_logits = logits[0]
            target_idx = y[i]

            # Softmax cross-entropy
            pred_probs = softmax(pred_logits[np.newaxis, :])[0]
            loss = -np.log(pred_probs[target_idx] + 1e-10)
            epoch_loss += loss

            # Simple gradient update (not proper backprop, just weight adjustment)
            if loss > 0.5:  # High loss - adjust weights
                W_q *= 1 - learning_rate * 0.1
                W_k *= 1 - learning_rate * 0.1
                W_v *= 1 - learning_rate * 0.1

        avg_loss = epoch_loss / num_samples
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

    trained_weights = {
        "W_q": W_q,
        "W_k": W_k,
        "W_v": W_v,
        "W_o": W_o,
    }

    return trained_weights


def save_weights(weights, filename="trained_weights.pkl"):
    """Save trained weights to file."""
    with open(filename, "wb") as f:
        pickle.dump(weights, f)
    print(f"Weights saved to {filename}")


def load_weights(filename="trained_weights.pkl"):
    """Load trained weights from file."""
    try:
        with open(filename, "rb") as f:
            weights = pickle.load(f)
        print(f"Weights loaded from {filename}")
        return weights
    except FileNotFoundError:
        print(f"File {filename} not found. Returning None.")
        return None


if __name__ == "__main__":
    # Train the model
    trained_weights = train_transformer(
        d_model=512,
        num_heads=4,
        num_epochs=50,
        learning_rate=0.001,
        num_samples=200,
    )

    # Save weights
    save_weights(trained_weights, "trained_weights.pkl")
    print("Training complete!")
