# BlossomModel

Shaun Harker. 2026-06-12.

[work in progress. to be tested.]

## Overview

The Blossom model is a transformer variant that processes input sequences in contiguous chunks called *blocks*. To reduce memory usage, it applies gradient checkpointing across these blocks. In this architecture, alternating layers cross-attend to the final output of previous blocks rather than generating their own local keys and values.

## Motivation

### The Structural Bound of Standard Attention
To understand the theoretical motivation behind the Blossom architecture, we must look at the topology of memory in standard Large Language Models.

Transformers maintain a high-dimensional memory across time via the Key-Value (KV) cache. However, this memory routing is horizontal. When Layer 12 processes a token, it attends to the Layer 12 representations of past tokens. When Layer 80 processes a token, it attends to past Layer 80 representations.

This creates an architectural bound: the maximum computational depth applied to any single forward pass is limited by the physical layer count (`num_layers`). The deepest representations computed at the final layer cannot directly inform the foundational processing of the next tokens in continuous high-dimensional space. Instead, that representation must be projected into a discrete vocabulary space (the tokenization bottleneck), fed back into the bottom of the network, and the process of building abstraction starts over from layer zero.

### The Proposed Mechanism: Recurrent Cross-Attention
The Blossom architecture proposes a topological change to test a specific hypothesis: What happens if we allow attention to route backward *and up*? Specifically, we create a final layer of information—our so-called *blossoms*—which lower layers in subsequent tokens may attend to.

Unfortunately, implementing such a strategy natively can bottleneck pretraining efficiency due to the sequential dependency. To mitigate this, we restrict the cross-attention to operate at a lag, enabling us to process blocks of contiguous tokens in parallel.

### Expanding Latent Depth Beyond `num_layers`
From a research perspective, this design is compelling because it removes the `num_layers` bound on computational depth. 

If a model can route the deep representations from Block 1 into the foundational layers of Block 2, it establishes a recurrent latent pathway. A representation could be transformed by 80 layers, passed as a continuous vector into the early layers of the next block, transformed 70 more times, and so on. 

This introduces the capacity for high-dimensional paths that are deeper than the network itself. The model gains a mechanism to carry continuous mathematical states across multiple sequence steps without being forced to articulate them as discrete text tokens.

---

## Architecture

### Sequence Blocks
Instead of processing an entire sequence at once, the model divides the input into smaller, sequential segments of length `block_size` (e.g., 32 tokens). The model processes these blocks sequentially from left to right. This block-wise processing, combined with gradient checkpointing, significantly minimizes memory consumption because intermediate activations within a block can be discarded and rematerialized later during the backward pass.

### Components

We use the following architectural components:

* RMSNorm
* SwiGLU Activations
* Grouped Query Attention (GQA)
* RoPE (Rotary Positional Embeddings)
* Bias-Free Linear Layers

### Layer Types
This architecture uses three specific layer types:

1. **`AttentionLayer`**: A standard self-attention layer utilizing GQA. It projects Q, K, and V, applies RoPE, reads past KVs from its local cache, writes its newly computed KVs, and performs self-attention.
2. **`CrossAttentionLayer`**: A layer that skips Key/Value generation. It computes Queries, applies RoPE, and cross-attends to a shared, global KV cache generated entirely by previous sequence blocks.
3. **`BlossomLayer`**: A layer executed strictly after the final transformer layer for a given sequence block. It projects the block's final hidden state into new, shared Keys and Values, applies RoPE, and writes them to the global cache so they can be consumed by subsequent sequence blocks.

### The Lag Constraint
Because the global Blossom KVs for a given sequence block are computed *after* that block passes through all preceding transformer layers, a query currently being processed in a block cannot attend to Blossom KVs from its own block. 

To safely accommodate this restriction and keep the attention mechanism mathematically consistent, we apply a lag constraint: queries only attend to blossoms that are at least `block_size` tokens in the past. This guarantees they are looking at completed KVs from previous blocks.

For the first block, the `CrossAttentionLayer` has no past blossoms to attend to, so it skips the attention step entirely and only executes the SwiGLU block.

### Differentiable Cache
To make gradient checkpointing work while processing blocks sequentially, the model must maintain a key-value (KV) cache during the forward training pass. Standard KV caches break PyTorch's autograd graph when gradient checkpointing is used. 

To address this, the Blossom model implements a custom differentiable cache using pre-allocated memory buffers and autograd hooks:
* **`DifferentiableCache`**: A pre-allocated tensor buffer for storing KVs.
* **`CacheRead` & `CacheWrite`**: Custom `torch.autograd.Function` hooks that dictate how gradients flow into and out of the cache.
* **Tracker Tensor**: A scalar tensor passed sequentially between reads and writes. This creates a topological dependency in the PyTorch autograd graph, ensuring that gradients accumulate correctly across different sequence blocks.
