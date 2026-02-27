import numpy as np
import matplotlib.pyplot as plt


# Function definitions - used in app.py
def softmax(x):
    """Compute softmax along last dimension"""
    # Subtract max for numerical stability (prevents overflow)
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    # Normalize to probability distribution (sum = 1)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Single attention head

    Args:
        Q: queries (seq_len, d_k)
        K: keys (seq_len, d_k)
        V: values (seq_len, d_v)
        mask: optional mask (seq_len, seq_len)

    Returns:
        output: (seq_len, d_v)
        attention_weights: (seq_len, seq_len)
    """
    # Compute attention scores: similarity between queries and keys
    d_k = Q.shape[-1]
    scores = (Q @ K.T) / np.sqrt(d_k)  # Scale by sqrt(d_k) for stable gradients
    scores = scores * 2.0  # Increase sharpness of attention

    # Apply mask if provided (e.g., prevent attending to future positions)
    if mask is not None:
        scores = scores + mask

    # Convert scores to attention weights (probabilities)
    attention_weights = softmax(scores)
    # Weighted sum of values using attention weights
    output = attention_weights @ V

    return output, attention_weights


def multi_head_attention(X, d_model, num_heads, mask=None, pretrained_weights=None):
    """
    Multi-head attention mechanism

    Args:
        X: input sequence (seq_len, d_model)
        d_model: total dimension
        num_heads: number of attention heads
        mask: optional attention mask (seq_len, seq_len)
        pretrained_weights: optional dict with pre-trained W_q, W_k, W_v, W_o

    Returns:
        output: (seq_len, d_model)
        all_attention_weights: list of (seq_len, seq_len) for each head
    """
    seq_len, _ = X.shape

    # Ensure d_model is divisible by num_heads
    assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

    d_k = d_model // num_heads  # Each head operates on smaller dimension

    # Use pre-trained weights if provided, otherwise initialize randomly
    if pretrained_weights is not None:
        W_q = pretrained_weights.get("W_q", np.random.randn(d_model, d_model) * 0.1)
        W_k = pretrained_weights.get("W_k", np.random.randn(d_model, d_model) * 0.1)
        W_v = pretrained_weights.get("W_v", np.random.randn(d_model, d_model) * 0.1)
        W_o = pretrained_weights.get("W_o", np.random.randn(d_model, d_model) * 0.1)
    else:
        # Projection matrices (learned in real models)
        W_q = np.random.randn(d_model, d_model) * 0.1
        W_k = np.random.randn(d_model, d_model) * 0.1
        W_v = np.random.randn(d_model, d_model) * 0.1
        W_o = np.random.randn(d_model, d_model) * 0.1

    # Project all heads at once (efficient)
    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v

    # Split into multiple heads: (seq_len, d_model) -> (seq_len, num_heads, d_k)
    Q = Q.reshape(seq_len, num_heads, d_k)
    K = K.reshape(seq_len, num_heads, d_k)
    V = V.reshape(seq_len, num_heads, d_k)

    # Rearrange for iteration: (seq_len, num_heads, d_k) -> (num_heads, seq_len, d_k)
    Q = Q.transpose(1, 0, 2)
    K = K.transpose(1, 0, 2)
    V = V.transpose(1, 0, 2)

    # Process each head independently
    head_outputs = []
    all_attention_weights = []

    for i in range(num_heads):
        head_output, attn_weights = scaled_dot_product_attention(
            Q[i], K[i], V[i], mask=mask
        )
        head_outputs.append(head_output)
        all_attention_weights.append(attn_weights)

    # Merge all heads back: (num_heads, seq_len, d_k) -> (seq_len, num_heads, d_k)
    concat_heads = np.stack(head_outputs, axis=1)
    # Flatten to original dimension: (seq_len, num_heads*d_k) -> (seq_len, d_model)
    concat_heads = concat_heads.reshape(seq_len, d_model)

    # Linear projection to combine head information
    output = concat_heads @ W_o

    return output, all_attention_weights


def positional_encoding(seq_len, d_model):
    """
    Generate positional encoding - encodes token position as waves

    Args:
        seq_len: length of sequence
        d_model: dimension of embeddings

    Returns:
        pos_encoding: (seq_len, d_model)
    """
    pos_encoding = np.zeros((seq_len, d_model))

    # Position indices for each token
    position = np.arange(seq_len)[:, np.newaxis]  # Column vector: [0, 1, 2, ...]

    # Frequency factors: decrease exponentially (different wavelengths per dimension)
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))

    # Sine waves for even dimensions
    pos_encoding[:, 0::2] = np.sin(position * div_term)
    # Cosine waves for odd dimensions
    pos_encoding[:, 1::2] = np.cos(position * div_term)

    return pos_encoding


def layer_norm(x, epsilon=1e-6):
    """
    Layer normalization - stabilizes values and training

    Args:
        x: input (seq_len, d_model)
        epsilon: small constant for numerical stability

    Returns:
        normalized: (seq_len, d_model)
    """
    # Normalize per sequence position to mean=0, std=1
    mean = np.mean(x, axis=-1, keepdims=True)
    std = np.std(x, axis=-1, keepdims=True)
    # Add epsilon to prevent division by zero
    return (x - mean) / (std + epsilon)


def feed_forward(x, d_model, d_ff):
    """
    Position-wise feed-forward network: expand, activate, contract

    Args:
        x: input (seq_len, d_model)
        d_model: model dimension
        d_ff: hidden dimension (typically 4 * d_model)

    Returns:
        output: (seq_len, d_model)
    """
    W1 = np.random.randn(d_model, d_ff) * 0.01
    b1 = np.zeros(d_ff)
    W2 = np.random.randn(d_ff, d_model) * 0.01
    b2 = np.zeros(d_model)

    # Expand and apply ReLU nonlinearity
    hidden = np.maximum(0, x @ W1 + b1)
    # Contract back to original dimension
    output = hidden @ W2 + b2

    return output


def encoder_layer(x, d_model, num_heads, d_ff):
    """
    Single Transformer encoder layer: Attention -> Norm -> FFN -> Norm

    Args:
        x: input (seq_len, d_model)
        d_model: model dimension
        num_heads: number of attention heads
        d_ff: feed-forward hidden dimension

    Returns:
        output: (seq_len, d_model)
    """
    # Mix information across all positions
    attn_output, _ = multi_head_attention(x, d_model, num_heads)
    # Residual connection + stabilize
    x = layer_norm(x + attn_output)

    # Process each position independently
    ff_output = feed_forward(x, d_model, d_ff)
    # Residual connection + stabilize
    output = layer_norm(x + ff_output)

    return output


def transformer_encoder(x, num_layers, d_model, num_heads, d_ff):
    """
    Stack of encoder layers - progressively refine the representation

    Args:
        x: input with positional encoding (seq_len, d_model)
        num_layers: number of encoder layers to stack
        d_model: model dimension
        num_heads: number of attention heads
        d_ff: feed-forward hidden dimension

    Returns:
        output: (seq_len, d_model)
    """
    # Each layer builds on previous: layer_i input = layer_(i-1) output
    for i in range(num_layers):
        x = encoder_layer(x, d_model, num_heads, d_ff)

    return x


def create_look_ahead_mask(seq_len):
    """
    Create mask to prevent attending to future positions (autoregressive)

    Args:
        seq_len: sequence length

    Returns:
        mask: (seq_len, seq_len) with -inf in upper triangle
    """
    # Upper triangle (future) = -inf (masked), lower triangle (past) = 0 (allowed)
    mask = np.triu(np.ones((seq_len, seq_len)) * -1e9, k=1)
    return mask


def cross_attention(decoder_input, encoder_output, d_model, num_heads):
    """
    Cross-attention: decoder (Q) attends to encoder (K,V) - bridges encoder/decoder

    Args:
        decoder_input: queries from decoder (seq_len_dec, d_model)
        encoder_output: keys/values from encoder (seq_len_enc, d_model)
        d_model: model dimension
        num_heads: number of attention heads

    Returns:
        output: (seq_len_dec, d_model)
        attention_weights: list of attention matrices
    """
    seq_len_dec, _ = decoder_input.shape
    seq_len_enc, _ = encoder_output.shape

    assert d_model % num_heads == 0
    d_k = d_model // num_heads

    # Different projection matrices for decoder queries vs encoder keys/values
    W_q = np.random.randn(d_model, d_model) * 0.01  # Decoder -> queries
    W_k = np.random.randn(d_model, d_model) * 0.01  # Encoder -> keys
    W_v = np.random.randn(d_model, d_model) * 0.01  # Encoder -> values
    W_o = np.random.randn(d_model, d_model) * 0.01

    # Q comes from decoder (what to generate), K,V from encoder (what's available)
    Q = decoder_input @ W_q
    K = encoder_output @ W_k
    V = encoder_output @ W_v

    # Split into heads (note: encoder and decoder can have different seq_len)
    Q = Q.reshape(seq_len_dec, num_heads, d_k).transpose(1, 0, 2)
    K = K.reshape(seq_len_enc, num_heads, d_k).transpose(1, 0, 2)
    V = V.reshape(seq_len_enc, num_heads, d_k).transpose(1, 0, 2)

    # Decoder positions attend to encoder positions
    head_outputs = []
    all_attention_weights = []

    for i in range(num_heads):
        head_output, attn_weights = scaled_dot_product_attention(Q[i], K[i], V[i])
        head_outputs.append(head_output)
        all_attention_weights.append(attn_weights)

    # Merge heads and project
    concat_heads = np.stack(head_outputs, axis=1)
    concat_heads = concat_heads.reshape(seq_len_dec, d_model)
    output = concat_heads @ W_o

    return output, all_attention_weights


def decoder_layer(x, encoder_output, d_model, num_heads, d_ff):
    """
    Single Transformer decoder layer: Masked SelfAttn -> CrossAttn -> FFN

    Args:
        x: decoder input (seq_len, d_model)
        encoder_output: output from encoder (seq_len_enc, d_model)
        d_model: model dimension
        num_heads: number of attention heads
        d_ff: feed-forward hidden dimension

    Returns:
        output: (seq_len, d_model)
    """
    seq_len = x.shape[0]

    # 1. Masked self-attention (can only see past, not future)
    mask = create_look_ahead_mask(seq_len)

    W_q = np.random.randn(d_model, d_model) * 0.01
    W_k = np.random.randn(d_model, d_model) * 0.01
    W_v = np.random.randn(d_model, d_model) * 0.01
    W_o = np.random.randn(d_model, d_model) * 0.01

    d_k = d_model // num_heads

    Q = (x @ W_q).reshape(seq_len, num_heads, d_k).transpose(1, 0, 2)
    K = (x @ W_k).reshape(seq_len, num_heads, d_k).transpose(1, 0, 2)
    V = (x @ W_v).reshape(seq_len, num_heads, d_k).transpose(1, 0, 2)

    head_outputs = []
    for i in range(num_heads):
        head_output, _ = scaled_dot_product_attention(Q[i], K[i], V[i], mask=mask)
        head_outputs.append(head_output)

    masked_attn_output = np.stack(head_outputs, axis=1).reshape(seq_len, d_model) @ W_o
    x = layer_norm(x + masked_attn_output)

    # 2. Cross-attention to encoder (attend to input encoding)
    cross_attn_output, _ = cross_attention(x, encoder_output, d_model, num_heads)
    x = layer_norm(x + cross_attn_output)

    # 3. Feed-forward and normalize
    ff_output = feed_forward(x, d_model, d_ff)
    output = layer_norm(x + ff_output)

    return output


def transformer_decoder(x, encoder_output, num_layers, d_model, num_heads, d_ff):
    """
    Stack of decoder layers - progressively generate and refine output

    Args:
        x: decoder input with positional encoding (seq_len, d_model)
        encoder_output: output from encoder (seq_len_enc, d_model)
        num_layers: number of decoder layers
        d_model: model dimension
        num_heads: number of attention heads
        d_ff: feed-forward hidden dimension

    Returns:
        output: (seq_len, d_model)
    """
    # Each layer refines decoder output while attending to encoder
    for i in range(num_layers):
        x = decoder_layer(x, encoder_output, d_model, num_heads, d_ff)

    return x
