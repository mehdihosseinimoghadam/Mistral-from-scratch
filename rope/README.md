# 🌟 RoPE: Rotary Positional Encoding Implementation

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> **Complete implementation of Rotary Positional Encoding (RoPE) - the technique powering modern language models like GPT-4, LLaMA, ChatGPT, and Mistral 7B!**

[![YouTube Tutorial](https://img.shields.io/badge/YouTube-Tutorial-red?logo=youtube)](https://youtube.com/watch?v=YOUR_VIDEO_ID)

**🎥 [Watch the Full Tutorial on YouTube](https://youtube.com/watch?v=YOUR_VIDEO_ID)**



## 🎯 What is RoPE?

**Rotary Positional Encoding (RoPE)** is a method to inject positional information into transformers by rotating query and key vectors in multi-head attention. Unlike traditional additive positional encoding, RoPE uses **multiplicative rotations** that provide:

- ✅ **Better extrapolation** to longer sequences
- ✅ **Maintained relative positions** through geometric rotation  
- ✅ **Improved context handling** compared to sinusoidal encoding
- ✅ **Mathematical elegance** using complex exponentials

## 🔥 Key Features

### 🆚 RoPE vs Traditional Positional Encoding

| Aspect | Traditional (Sinusoidal) | RoPE (Rotary) |
|--------|-------------------------|---------------|
| **Type** | Additive (added to embeddings) | Multiplicative (applied via rotation) |
| **Implementation** | Add sinusoidal vectors | Rotate Q/K vectors using complex exponentials |
| **Extrapolation** | Limited beyond training length | Better generalization to unseen positions |
| **Context Scaling** | Poor beyond training length | Scales better to longer sequences |
| **Mathematical Form** | `e_pos = sin/cos(pos · freq)` | `RoPE(x,pos) = R(pos) · x` |
| **Used In** | Original Transformer | GPT-NeoX, GPT-J, LLaMA, ChatGPT |

## 🏗️ Architecture Integration

RoPE seamlessly integrates into the **self-attention mechanism**:

```
Input Embeddings → RMS Norm → Self-Attention (with RoPE) → Feed Forward → Output
                                    ↓
                    Q_rot, K_rot ← RoPE(Q, K, position)
```

## 📊 Mathematical Foundation

### Complex Rotation
```math
R_θ(z) = z · e^{iθ} = z · (cos θ + i sin θ)
```

### 2D Rotation Matrix
```math
R_θ = [cos θ  -sin θ]
      [sin θ   cos θ]
```

### RoPE Formula
```math
RoPE(x_i, pos) = R(pos · θ_i) · x_i
```

Where `θ_i = 10000^(-2i/d)` for dimension pairs.




### Memory Optimization

- **Precomputed frequencies**: Stored once and reused
- **In-place rotations**: Minimal memory overhead
- **Block-diagonal structure**: Leverages sparsity



## 📊 Models Using RoPE

| Model | Organization | Parameters | Context Length |
|-------|-------------|------------|----------------|
| **GPT-J** | EleutherAI | 6B | 2048 |
| **GPT-NeoX** | EleutherAI | 20B | 2048 |
| **LLaMA** | Meta | 7B-65B | 2048-4096 |
| **ChatGPT** | OpenAI | Unknown | 4096+ |
| **Mistral 7B** | Mistral AI | 7B | 8192 |

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md).



## 📚 References & Research

### 📄 Key Papers
- **RoFormer: Enhanced Transformer with Rotary Position Embedding** - Su et al. (2021)
- **Attention Is All You Need** - Vaswani et al. (2017)
- **GPT-J-6B: A 6 Billion Parameter Autoregressive Language Model** - EleutherAI
- **LLaMA: Open and Efficient Foundation Language Models** - Touvron et al. (2023)

### 🔗 Useful Links
- [Original RoFormer Paper](https://arxiv.org/abs/2104.09864)
- [EleutherAI GPT-J Implementation](https://github.com/kingoflolz/mesh-transformer-jax)
- [LLaMA Model Card](https://github.com/facebookresearch/llama)
- [Mistral 7B Technical Report](https://arxiv.org/abs/2310.06825)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## 📞 Support & Contact

---

⭐ **If this implementation helped you, please give it a star!** ⭐

![RoPE Visualization](images/rope_visualization.png)

*RoPE rotation visualization showing how position encoding is applied through geometric rotation in the embedding space.*
