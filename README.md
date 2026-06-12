# BlossomModel

[work in progress. to be tested.]

The Blossom model is a transformer variant that processes input sequences in contiguous chunks called *blocks*. To reduce memory usage, it applies gradient checkpointing across these blocks. In this architecture, alternating layers cross-attend to the final output of previous blocks rather than generating their own local keys and values.

## Sequence Blocks
Instead of processing an entire sequence at once, the model divides the input into smaller, sequential segments of length `block_size` (e.g., 32 tokens). The model processes these blocks sequentially from left to right. This block-wise processing, combined with gradient checkpointing, significantly minimizes memory consumption because intermediate activations within a block can be discarded and rematerialized later during the backward pass.

## Modern Architectural Standards (2026)

This model has been written to conform to the dominant hardware-efficient architectural standards seen in modern frontier models (like LLaMA 3, Gemma 4, and DeepSeek V2):

* **RMSNorm:** Replaces classic `LayerNorm`. By dropping the mean-centering step, we save compute overhead while retaining total mathematical stability.
* **SwiGLU Activations:** Replaces standard GELU-based Feed-Forward Networks. The `mlp_ratio` natively adjusts the hidden dimension size (defaulting to `8/3` to replicate the parameter count of a classic `4x` expansion width). 
* **Grouped Query Attention (GQA):** Controlled via `kv_heads`. GQA massively reduces memory allocations for keys and values, which is an absolute necessity for accommodating long sequences.
* **RoPE (Rotary Positional Embeddings):** An implicit relative positional encoding applied independently to Queries and Keys to natively encapsulate token distances.
* **Bias-Free Linear Layers:** Following LLaMA protocols, all bias vectors have been removed from `nn.Linear` projection pathways to marginally increase training throughput and streamline memory layouts.

## Layer Types

This architecture uses three specific layer types:

1. **`AttentionLayer`**: A standard self-attention layer utilizing GQA. It projects Q, K, and V, applies RoPE, reads past KVs from its local cache, writes its newly computed KVs, and performs self-attention.
2. **`CrossAttentionLayer`**: A layer that skips Key/Value generation. It computes Queries, applies RoPE, and cross-attends to a shared, global KV cache generated entirely by previous sequence blocks.
3. **`BlossomLayer`**: A layer executed strictly after the final transformer layer for a given sequence block. It projects the block's final hidden state into new, shared Keys and Values, applies RoPE, and writes them to the global cache so they can be consumed by subsequent sequence blocks.

### The Lag Constraint

Because the global Blossom KVs for a given sequence block are computed *after* that block passes through all preceding transformer layers, a query currently being processed in a block cannot attend to Blossom KVs from its own block. 

To safely accommodate this restriction and keep the attention mechanism mathematically consistent, we apply a lag constraint: queries only attend to blossoms that are at least `block_size` tokens in the past. This guarantees they are looking at completed KVs from previous blocks.

For the first block, the `CrossAttentionLayer` has no past blossoms to attend to, so it skips the attention step entirely and only executes the SwiGLU block.

## Differentiable Cache

To make gradient checkpointing work while processing blocks sequentially, the model must maintain a key-value (KV) cache during the forward training pass. Standard KV caches break PyTorch's autograd graph when gradient checkpointing is used. 

To address this, the Blossom model implements a custom differentiable cache using pre-allocated memory buffers and autograd hooks:
* **`DifferentiableCache`**: A pre-allocated tensor buffer for storing KVs.
* **`CacheRead` & `CacheWrite`**: Custom `torch.autograd.Function` hooks that dictate how gradients flow into and out of the cache.
* **Tracker Tensor**: A scalar tensor passed sequentially between reads and writes. This creates a topological dependency in the PyTorch autograd graph, ensuring that gradients accumulate correctly across different sequence blocks.

## Usage

Initialize the model and pass inputs and targets as standard tensors. Play with the `kv_heads` param to see the impact of Grouped Query Attention.

```python
model = BlossomModel(
    vocab_size=32000, 
    dim=512, 
    heads=8,
    kv_heads=4,      # GQA: 4 KV heads to 8 Query heads (2:1 ratio) 
    num_layers=6, 
    lag=32,          # Must be >= block_size
    block_size=32,   # Size of the sequence blocks for gradient checkpointing
    max_seq_len=8192 # Required for RoPE precomputation
).cuda()

# (Batch, Seq_Len)
input_ids = torch.randint(0, 32000, (4, 128)).cuda()
targets = torch.randint(0, 32000, (4, 128)).cuda()

loss = model(input_ids, targets)
loss.backward()