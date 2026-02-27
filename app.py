import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

# Import your transformer functions here
from transformer import positional_encoding, multi_head_attention

st.set_page_config(page_title="Transformer Mad Libs", page_icon="🤖", layout="wide")


# Cache the weights loading to avoid reloading on every interaction
@st.cache_data
def load_trained_weights():
    """Load and cache trained weights to avoid repeated disk I/O."""
    if os.path.exists("trained_weights.pkl"):
        with open("trained_weights.pkl", "rb") as f:
            return pickle.load(f)
    return None


# Cache positional encoding calculation
@st.cache_data
def get_positional_encoding(seq_len, d_model):
    """Cache positional encoding to avoid recalculation."""
    return positional_encoding(seq_len, d_model)


st.title("🤖 Transformer Mad Libs Visualiser")
st.markdown("*See how Transformers 'pay attention' to words in your sentence*")

# Sidebar for technical controls
with st.sidebar:
    st.header("⚙️ Transformer Settings")
    d_model = 512
    st.markdown(f"**Model dimension:** {d_model}")

    # Calculate valid head options (divisors of d_model, up to 16)
    valid_heads = [i for i in range(1, min(17, d_model + 1)) if d_model % i == 0]
    num_heads = st.selectbox(
        "Number of attention heads", valid_heads, index=valid_heads.index(4)
    )
    st.info(
        f"💡 The number of attention heads must divide equally into the model dimension ({d_model})."
    )
    st.markdown("---")
    st.markdown("**What you're seeing:**")
    st.markdown("- Darker colours = stronger attention")
    st.markdown("- Each head focuses on different patterns")
    st.markdown("- Multiple heads = better understanding")

    st.markdown("---")
    st.markdown("**Advanced Features**")
    use_causal_mask = st.checkbox("Apply Causal Masking", value=False)
    if use_causal_mask:
        st.info(
            "🔐 **Causal Masking:** Words can only attend to previous words (used in autoregressive models like GPT). "
            'This prevents the model from "cheating" by looking ahead to future words during text generation. '
            "You'll see a triangular pattern in the attention matrix - only the lower triangle is active."
        )

    # Load trained weights
    trained_weights = load_trained_weights()

    use_trained = False
    if trained_weights is not None:
        use_trained = st.checkbox("Use Trained Weights", value=False)
        if use_trained:
            st.success("✅ Using trained model weights!")
            st.info(
                "📚 **Trained vs Random:**\n\n"
                "With trained weights, the model has learned meaningful attention patterns. "
                "Compare to random weights to see the difference training makes. "
                "Notice how trained weights create richer, more varied attention patterns!"
            )
    else:
        st.warning("⚠️ Trained weights not available (run train_transformer.py first)")

# Sentence input
st.subheader("📝 Build Your Sentence")

sentence_mode = st.radio(
    "Choose input method:",
    [
        "Structured Sentence (simpler and easier to understand)",
        "Custom Sentence (have some fun!)",
    ],
    horizontal=True,
)

if sentence_mode == "Structured Sentence (simpler and easier to understand)":
    col1, col2 = st.columns(2)
    with col1:
        adj1 = st.text_input("Adjective 1", "sleepy", max_chars=20)
        noun1 = st.text_input("Noun 1", "wizard", max_chars=20)
        verb = st.text_input("Past tense verb", "danced", max_chars=20)
    with col2:
        adj2 = st.text_input("Adjective 2", "purple", max_chars=20)
        noun2 = st.text_input("Noun 2", "elephant", max_chars=20)

    sentence_structure = f"The {adj1} {noun1} {verb} at the {adj2} {noun2}"
else:
    user_sentence = st.text_area(
        "Type any sentence to visualise attention patterns:",
        value="The sleepy wizard danced at the purple elephant",
        max_chars=75,
        height=60,
    )
    sentence_structure = user_sentence.strip()

# Tokenize the sentence
tokens = sentence_structure.split()

st.markdown("### Your Sentence:")
st.markdown(f'**"{sentence_structure}"**')

# Generate embeddings and run attention BEFORE columns
seq_len = len(tokens)
np.random.seed(42)  # For consistency
X = np.random.randn(seq_len, d_model)

# Add positional encoding
pe = get_positional_encoding(seq_len, d_model)
X_with_pos = X + pe

# Create causal mask if enabled
mask = None
if use_causal_mask:
    # Create lower triangular mask (can attend to current and past, not future)
    mask = np.triu(np.full((seq_len, seq_len), -np.inf), k=1)

# Run multi-head attention
output, attention_weights = multi_head_attention(
    X_with_pos,
    d_model,
    num_heads,
    mask=mask,
    pretrained_weights=trained_weights if use_trained else None,
)

# Now create layout columns for display
st.markdown("---")
info_col, attention_col = st.columns([1, 2])

with info_col:
    st.markdown("### 🔍 How to Read the Attention Patterns")
    st.markdown(
        """
Each cell shows how much one word "attends to" another word.
- Row = word asking "what should I pay attention to?"
- Column = word being attended to
- Darker = stronger connection
"""
    )

    st.markdown("---")
    st.markdown("### 💡 What Does This Mean?")

    # Head selection for interpretation
    head_idx = st.selectbox(
        "Select attention head to visualise",
        range(num_heads),
        format_func=lambda x: f"Head {x+1}",
        key="head_selector",
    )

    # Find highest attention weights for interesting insights
    max_attention = np.max(attention_weights[head_idx])
    max_pos = np.unravel_index(
        np.argmax(attention_weights[head_idx]), attention_weights[head_idx].shape
    )

    st.markdown(
        f"""
In **Head {head_idx + 1}**, the word **"{tokens[max_pos[0]]}"** pays the most attention to **"{tokens[max_pos[1]]}"** 
(weight: {max_attention:.2f}).

This means when the model processes "{tokens[max_pos[0]]}", it's strongly considering 
the meaning of "{tokens[max_pos[1]]}" to understand the sentence.
"""
    )

    st.markdown("---")
    st.markdown("### 📈 Trends to Observe")

    trends_text = """
Look for these patterns across different heads:
- **Adjectives** often attend to the nouns they modify
- **Verbs** connect to their subjects and objects
- **Prepositions** link related concepts
- Different heads specialise in different relationship types
"""

    if use_trained:
        trends_text += """

**With Trained Weights:**
- **Rich patterns:** The model has learned which words are important for context
- **Varied attention:** Different heads focus on different linguistic relationships
- **Meaningful structure:** Attention weights reflect actual grammatical and semantic relationships
- **Comparison insight:** Compare to random weights to see how much training matters!
"""
    else:
        trends_text += """

**With Random Weights (Untrained):**
- **Arbitrary patterns:** The model hasn't learned anything yet - attention is random
- **Self-attention dominance:** Words tend to attend to themselves most (especially with masking)
- **No clear structure:** Patterns don't reflect meaningful linguistic relationships
- **Why it matters:** This shows that transformer architecture alone isn't enough - **training is essential**!
"""

    if use_causal_mask:
        trends_text += """

**With Causal Masking Enabled:**
- **Triangular pattern:** Only the lower-right triangle is active (words attending to themselves and previous words)
- **No future attention:** Each word CANNOT attend to words that come after it
- **Why it matters:** This is essential for text generation - the model generates words one at a time, so it can't "peek" at future words it hasn't generated yet
- **GPT uses this:** Models like GPT rely on causal masking to generate coherent text sequentially

**⚠️ Note on Self-Attention Dominance:**
You may notice words attending mostly to themselves. This is normal with random initialisation - without training, the model hasn't learned meaningful patterns yet. In a trained model (like GPT), you'd see rich attention to previous context. This demonstrates why **training is crucial** - it teaches the model what patterns matter!
"""

    st.markdown(trends_text)

with attention_col:
    st.subheader("🔍 Attention Head Visualiser")

    # Create heatmap using the head_idx selected in sentence column
    fig, ax = plt.subplots(figsize=(10, 8))

    # Prepare attention weights and labels
    attn_data = attention_weights[head_idx]
    xticklabels = list(tokens)
    yticklabels = list(tokens)

    # If causal masking is enabled, reverse the "Attending TO" axis to show triangle clearly
    if use_causal_mask:
        attn_data = attn_data[:, ::-1]  # Reverse columns
        xticklabels = tokens[::-1]  # Reverse token order
        # Keep y-axis in original sentence order for clarity

    sns.heatmap(
        attn_data,
        xticklabels=xticklabels,
        yticklabels=yticklabels,
        cmap="YlOrRd",
        annot=True,
        fmt=".2f",
        square=True,
        cbar_kws={"label": "Attention Weight"},
        ax=ax,
    )
    plt.title(
        f"Attention Head {head_idx + 1}: How Words Attend to Each Other",
        fontsize=14,
        fontweight="bold",
    )
    plt.xlabel("Attending TO (keys)", fontsize=12)
    plt.ylabel("Attending FROM (queries)", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    st.pyplot(fig)

# Footer section with interpretation and learning
st.markdown("---")
learning_col = st.container()

with learning_col:
    st.markdown("### 🎓 What You're Learning")
    st.markdown(
        """
This visualization shows **self-attention**, the core mechanism of Transformers (GPT, BERT, Claude).

**Key insights:**
1. Each word "queries" other words to understand context
2. Multiple heads capture different types of relationships
3. Attention weights determine which words influence each other
4. This is how AI understands language structure

Built with NumPy from scratch. No pre-trained models. [See the code →](https://github.com/JemHRice/transformer-visualise-app)
"""
    )

# Show all heads comparison
if st.checkbox("Show all heads at once"):
    st.subheader("All Attention Heads Comparison")

    fig, axes = plt.subplots(2, num_heads // 2, figsize=(20, 10))
    axes = axes.flatten()

    for i in range(num_heads):
        sns.heatmap(
            attention_weights[i],
            xticklabels=tokens,
            yticklabels=tokens,
            cmap="YlOrRd",
            square=True,
            cbar=False,
            ax=axes[i],
        )
        axes[i].set_title(f"Head {i+1}", fontsize=10)
        axes[i].tick_params(labelsize=8)

    plt.tight_layout()
    st.pyplot(fig)

    st.markdown(
        """
    **Notice:** Different heads focus on different patterns!
    - Some heads might connect nouns to their adjectives
    - Others might link verbs to their subjects
    - This is why multi-head attention is powerful
    """
    )

# Positional encoding visualization
with st.expander("📊 See Positional Encoding"):
    fig2, ax2 = plt.subplots(figsize=(12, 4))
    plt.pcolormesh(pe.T, cmap="RdBu", shading="auto")
    plt.xlabel("Position in Sentence", fontsize=12)
    plt.ylabel("Embedding Dimension", fontsize=12)
    plt.title("Positional Encoding Pattern", fontsize=14, fontweight="bold")
    plt.colorbar(label="Encoding Value")

    # Add word labels
    for i, token in enumerate(tokens):
        plt.text(i + 0.5, -5, token, ha="center", fontsize=10, rotation=45)

    plt.tight_layout()
    st.pyplot(fig2)

    st.markdown(
        """
    #### Why Positional Encoding Matters
    
    Transformers have **no inherent sense of word order** (unlike RNNs which process words sequentially).
    Positional encoding solves this by adding a unique pattern to each word based on its position in the sentence.
    
    #### What You're Seeing
    
    The heatmap above shows the positional encoding pattern for your sentence:
    - **Horizontal axis (X):** Position of each word in your sentence (left to right)
    - **Vertical axis (Y):** The 512 embedding dimensions (each word is represented by 512 numbers)
    - **Colour:** The value of each dimension (blue = negative, red = positive)
    
    Each column represents a unique word position and has a **different pattern** of values. This tells the transformer:
    - "This is the 1st word" vs "This is the 5th word" vs "This is the 10th word"
    
    #### The Pattern: Sine and Cosine Waves
    
    The encoding uses **sine and cosine waves** at different frequencies. However, due to the 512 dimensions being packed vertically, 
    the individual waves are **hard to see** in the heatmap - they blend together into the pattern you see above.
    
    **Why sine and cosine?**
    - They're **periodic**: Different positions produce different values, but they repeat in a predictable way
    - They're **smooth**: The model can learn "nearby positions are similar"
    - They're **scalable**: Work for sentences of any length
    - They have **diverse frequencies**: Different dimensions oscillate at different rates
    
    **Pattern breakdown:**
    - **Left side (low dimensions):** Long-wavelength sine/cosine - changes slowly across positions (coarse location info)
    - **Right side (high dimensions):** Short-wavelength sine/cosine - changes quickly across positions (fine positional details)
    
    This pattern is **the same every time** (using the same position), so it's deterministic and the model can learn to use it effectively.
    """
    )

# Fun examples
with st.expander("💡 Try These Fun Examples"):
    st.markdown(
        """
    **Silly sentences to try:**
    - The *invisible* *ninja* *farted* at the *confused* *banana*
    - The *ancient* *robot* *exploded* at the *tiny* *universe*
    - The *sparkly* *dinosaur* *laughed* at the *mysterious* *potato*
    """
    )
