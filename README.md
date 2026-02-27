# 🤖 Transformer Attention Visualiser

A visual, interactive tool for understanding how attention mechanisms work in transformers. Built by someone learning transformers, for everyone learning transformers.

### 🌐 **[Try the Live Demo!](https://transformer-visualise-app-akxrdapmcxbfelbunmzjr9.streamlit.app/)**

No installation needed—just click above to explore attention patterns interactively!

**⏱️ Note on Loading Time:**
- **First load**: Takes ~30-60 seconds (Streamlit Cloud server is starting up from cold—this is normal and one-time)
- **After that**: Much faster! ~2-5 seconds per interaction
- **On your own machine**: Loads instantly once installed locally

If the online demo feels slow, you can always **[run it locally](#option-2-run-locally)** for the best experience!

## 🎯 Why This Project?

I wanted to truly understand how transformers work, and I realized that **seeing is believing**. Reading about attention mechanisms is one thing, but *watching* how different words attend to each other across a sentence? That's where the magic clicks.

This tool lets you **build sentences and watch in real-time** how a transformer's attention heads focus on different patterns. It's designed to be intuitive and visual—perfect for learners like me who need to see the math come to life.

## ✨ Features

### 🎨 Interactive Sentence Building
- **Structured Mode**: Build sentences from scratch using adjectives, nouns, and verbs (great for beginners)
- **Custom Mode**: Type any sentence to visualise how the model attends to it

### 👁️ Attention Head Visualisation
- Visualise individual attention heads with colour-coded heatmaps
- Darker colours = stronger attention weights
- Watch how different heads focus on different patterns
- Select from 1-16 attention heads (configurable based on model dimension)

### 🔐 Causal Masking
- Toggle on to see how **autoregressive models** (like GPT) work
- Words can only attend to previous words—prevents "cheating" by looking ahead
- See the characteristic triangular pattern in the attention matrix
- Essential for understanding how language models generate text token-by-token

### 📚 Trained vs Random Weights
- **Compare trained weights** (model learned from data) vs **random weights** (untrained)
- See the dramatic difference training makes
- Trained weights show rich, meaningful attention patterns
- Random weights show why neural networks need training

### 📖 Detailed Explanations
- Learn about positional encoding and why transformers need it
- Understand multi-head attention and why multiple heads help
- See how causal masking prevents information leakage
- Interactive "Trends to Observe" section based on your current settings

## 🚀 Getting Started

### Option 1: Try Online (Easiest!)
👉 **[Open the live demo](https://transformer-visualise-app-akxrdapmcxbfelbunmzjr9.streamlit.app/)** – No installation needed!

### Option 2: Run Locally

**Prerequisites**
- Python 3.8+
- Virtual environment (recommended)

**Installation**

1. **Clone the repository**
   ```bash
   git clone https://github.com/JemHRice/transformer-visualise-app.git
   cd transformer-visualise-app
   ```

2. **Create and activate a virtual environment** (if you haven't already)
   ```bash
   python -m venv transformervenv
   transformervenv\Scripts\activate  # Windows
   # or
   source transformervenv/bin/activate  # Mac/Linux
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the app**
   ```bash
   streamlit run app.py
   ```

5. **Open in your browser**
   - Streamlit will open automatically, or navigate to `http://localhost:8501`
   - **✅ It will load instantly and be super responsive!** (Much faster than the cloud version)

### Training Your Own Weights (Optional)

Want to train your own model and see what trained weights look like? You can optionally install PyTorch and run the training script:

```bash
# Install PyTorch (CPU version)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Or for CUDA 11.8 (NVIDIA GPU)
pip install torch --index-url https://download.pytorch.org/whl/cu118

# Then train
python train_with_pytorch.py
```

This will:
- Train a transformer on sample English sentences
- Learn meaningful attention patterns through backpropagation
- Save weights to `trained_weights.pkl`
- Enable the "Use Trained Weights" toggle in the app

**Note:** PyTorch is not required to run the app—it's only needed if you want to train your own model locally.

## 📊 How to Use the App

### Step 1: Choose Your Sentence
- Select **Structured Sentence** to build from components (easier to understand)
- Or select **Custom Sentence** to type anything you want

### Step 2: Configure the Model
In the sidebar, you can:
- **Select number of attention heads** (must divide evenly into 512)
- **Enable causal masking** to simulate how GPT-like models work
- **Toggle trained weights** (if available) to see learned vs random patterns

### Step 3: Explore the Attention
- Move through different attention heads in the visualiser
- Watch how the colours change—darker = more attention
- Notice which words attend to which other words
- Compare patterns across different heads

### Step 4: Read the Explanations
- The middle column shows "Trends to Observe"—tips on what to look for
- Learn why certain patterns emerge
- Understanding deepens when you connect theory to what you see

## 🎓 Learning Resources

While building this tool, I benefited enormously from these resources. Check them out:

### 3Blue1Brown - Neural Networks Series
- **[Attention in Transformers, Visually Explained](https://www.youtube.com/watch?v=eMlx5aFJeuw)** - The clearest visual explanation of attention I've seen
- Great for building intuition before diving into the maths

### Andrej Karpathy - Attention Deep Dives
- **[Let's build GPT: from scratch, in code](https://www.youtube.com/watch?v=kCc8FmEb1nY)** - Beautiful walkthrough of building attention from first principles
- Watch Karpathy code attention in real-time—extremely educational

### Other Great Resources
- **[Attention is All You Need](https://arxiv.org/abs/1706.03762)** - The original transformer paper (dense but comprehensive)
- **[The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)** - Amazing blog post with diagrams

## 📁 Project Structure

```
transformer-from-scratch/
├── app.py                          # Streamlit web interface
├── transformer.py                  # Core attention mechanisms (NumPy)
├── train_with_pytorch.py           # Training script (PyTorch)
├── trained_weights.pkl             # Pre-trained weight matrices
├── requirements.txt                # Dependencies
└── README.md                       # This file
```

### Key Files Explained

**`app.py`**: The interactive Streamlit interface. This is what you see in the browser.

**`transformer.py`**: Pure NumPy implementation of:
- Positional encoding
- Scaled dot-product attention
- Multi-head attention

**`train_with_pytorch.py`**: PyTorch training script that learns weights from English sentences.

## 🔬 Under the Hood

### The Maths (Simplified)

**Attention is a similarity function:**
1. Convert each word to Query (Q), Key (K), and Value (V)
2. Compute similarity: `Attention(Q, K, V) = softmax(QK^T / √d) V`
3. Softmax ensures weights sum to 1 (probability distribution)
4. Multiply by values to get weighted combination

**Multi-head attention:**
- Process attention in parallel with different weights
- Each head learns different relationships
- Combine results for richer representations

**Causal masking:**
- Set future positions to `-∞` before softmax
- Forces attention weights to 0 for future tokens
- Essential for autoregressive generation

See the papers and videos above for the full mathematical derivation!

## 💡 What I Learned Building This

1. **Attention is learnable similarity** - The weights start random but learn what to pay attention to
2. **Multiple heads = multiple perspectives** - Different heads catch different patterns simultaneously
3. **Training matters** - The difference between trained and random weights is stark
4. **Causal masking prevents cheating** - In autoregressive models, this is what ensures tokens can't look ahead
5. **Visualisation is powerful** - A heatmap teaches better than a equation (at least for me!)

## 🛠️ Customisation Ideas

Want to extend this? Here are some ideas:

- Add **cross-attention** (how transformer encoders attend to decoders)
- Show **activation functions** and how they change representations
- Add **layer norm** and **residual connections** visualisation
- Train on **different datasets** (poetry, code, different languages)
- Compare **different architectures** side-by-side
- Add **ablation studies** (what happens if we remove causal masking?)

## 🤝 Let's Learn Together

This project is part of my learning journey. If you're also learning transformers and find this tool helpful, that makes me happy! If you find bugs or have ideas for improvements, feel free to explore and modify.

## 📝 License

This project is open for educational use. Feel free to learn from it, modify it, and build on it!

## 🙏 Acknowledgements

- Thanks to 3Blue1Brown for clarity and visual intuition
- Thanks to Andrej Karpathy for the detailed walkthroughs
- The transformer community for amazing resources
- Streamlit for making visualisation accessible

---

**Happy learning!** 🚀

Remember: transformers are just repeated applications of attention. Once you understand attention, you understand transformers. And now, you can *see* attention. 👀
