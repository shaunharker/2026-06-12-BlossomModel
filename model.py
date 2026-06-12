import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

# ---------------------------------------------------------
# 1. Modern Core Components (RMSNorm, SwiGLU, RoPE)
# ---------------------------------------------------------

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm_x = torch.mean(x.pow(2), dim=-1, keepdim=True)
        return x * torch.rsqrt(norm_x + self.eps) * self.weight

class SwiGLU(nn.Module):
    """Swish Gated Linear Unit, standard MLP replacement."""
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.w1 = nn.Linear(in_features, hidden_features, bias=False)
        self.w2 = nn.Linear(in_features, hidden_features, bias=False)
        self.w3 = nn.Linear(hidden_features, out_features, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

def precompute_freqs_cis(dim: int, max_seq_len: int, theta: float = 10000.0):
    """Precomputes complex frequencies for RoPE."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)

def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor):
    """Applies RoPE to a given Query or Key tensor."""
    # x shape: (Batch, Seq_Len, Heads, Head_Dim)
    # freqs_cis shape: (Seq_Len, Head_Dim // 2)
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.view(1, x_complex.shape[1], 1, x_complex.shape[-1])
    x_rotated = torch.view_as_real(x_complex * freqs_cis).flatten(3)
    return x_rotated.type_as(x)


# ---------------------------------------------------------
# 2. Differentiable KV Cache for Gradient Checkpointing
# ---------------------------------------------------------

class DifferentiableCache:
    """Pre-allocated memory buffer for storing KVs across sequence blocks."""
    def __init__(self, batch_size, max_seq_len, dim, device, dtype=torch.float32):
        self.data = torch.zeros((batch_size, max_seq_len, dim), dtype=dtype, device=device)
        self.grad = torch.zeros_like(self.data)

class CacheRead(torch.autograd.Function):
    """Reads from the cache up to the current sequence index."""
    @staticmethod
    def forward(ctx, cache_obj, read_end_idx, tracker):
        ctx.cache_obj = cache_obj
        ctx.read_end_idx = read_end_idx
        return cache_obj.data[:, :read_end_idx, :].clone(), tracker.clone()

    @staticmethod
    def backward(ctx, grad_out, grad_tracker):
        ctx.cache_obj.grad[:, :ctx.read_end_idx, :] += grad_out.detach()
        return None, None, grad_tracker

class CacheWrite(torch.autograd.Function):
    """Writes newly computed KVs for the current sequence block into the cache."""
    @staticmethod
    def forward(ctx, cache_obj, start_idx, end_idx, new_kv, tracker):
        ctx.cache_obj = cache_obj
        ctx.start_idx = start_idx
        ctx.end_idx = end_idx
        cache_obj.data[:, start_idx:end_idx, :] = new_kv.detach()
        return tracker.clone()

    @staticmethod
    def backward(ctx, grad_tracker):
        grad_new_kv = ctx.cache_obj.grad[:, ctx.start_idx:ctx.end_idx, :].clone()
        ctx.cache_obj.grad[:, ctx.start_idx:ctx.end_idx, :] = 0.0
        return None, None, None, grad_new_kv, grad_tracker


# ---------------------------------------------------------
# 3. Attention Layers
# ---------------------------------------------------------
class AttentionLayer(nn.Module):
    """
    A standard causal self-attention layer utilizing GQA, RoPE, and SwiGLU.
    It reads past KVs from its local cache, writes new KVs, and computes self-attention.
    """
    def __init__(self, dim, heads, kv_heads=None, mlp_ratio=8/3, norm_eps=1e-6):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.kv_heads = kv_heads if kv_heads is not None else heads
        self.head_dim = dim // heads
        
        self.q_proj = nn.Linear(dim, self.heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        
        self.norm1 = RMSNorm(dim, eps=norm_eps)
        self.norm2 = RMSNorm(dim, eps=norm_eps)
        
        hidden_dim = int(mlp_ratio * dim)
        self.mlp = SwiGLU(dim, hidden_dim, dim)

    def forward(self, x, start_idx, end_idx, cache_obj, tracker, freqs_cis):
        B, L_q, C = x.shape
        
        residual = x
        x = self.norm1(x)
        
        q = self.q_proj(x).view(B, L_q, self.heads, self.head_dim)
        k = self.k_proj(x).view(B, L_q, self.kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, L_q, self.kv_heads, self.head_dim)
        
        # Apply RoPE to Queries and Keys
        q = apply_rotary_emb(q, freqs_cis[start_idx:end_idx])
        k = apply_rotary_emb(k, freqs_cis[start_idx:end_idx])
        
        if start_idx > 0:
            past_kv, tracker = CacheRead.apply(cache_obj, start_idx, tracker)
            past_k, past_v = past_kv.chunk(2, dim=-1)
            
            past_k = past_k.view(B, start_idx, self.kv_heads, self.head_dim)
            past_v = past_v.view(B, start_idx, self.kv_heads, self.head_dim)
            
            full_k = torch.cat([past_k, k], dim=1)
            full_v = torch.cat([past_v, v], dim=1)
        else:
            full_k, full_v = k, v
            
        # Store flat tensors into cache
        local_kv = torch.cat([k.flatten(2), v.flatten(2)], dim=-1)
        tracker = CacheWrite.apply(cache_obj, start_idx, end_idx, local_kv, tracker)
        
        L_k = full_k.shape[1]
        
        # GQA Repeat Interleave Logic (if kv_heads < heads)
        if self.heads != self.kv_heads:
            n_rep = self.heads // self.kv_heads
            full_k = full_k.unsqueeze(3).expand(B, L_k, self.kv_heads, n_rep, self.head_dim).reshape(B, L_k, self.heads, self.head_dim)
            full_v = full_v.unsqueeze(3).expand(B, L_k, self.kv_heads, n_rep, self.head_dim).reshape(B, L_k, self.heads, self.head_dim)
        
        q = q.transpose(1, 2)
        full_k = full_k.transpose(1, 2)
        full_v = full_v.transpose(1, 2)
        
        q_pos = torch.arange(start_idx, end_idx, device=x.device).view(-1, 1)
        k_pos = torch.arange(0, L_k, device=x.device).view(1, -1)
        causal_mask = (k_pos <= q_pos).unsqueeze(0).unsqueeze(0)
        
        out = F.scaled_dot_product_attention(q, full_k, full_v, attn_mask=causal_mask)
        
        out = out.transpose(1, 2).reshape(B, L_q, C)
        x = residual + self.out_proj(out)
        x = x + self.mlp(self.norm2(x))
        return x, tracker


class CrossAttentionLayer(nn.Module):
    """
    A layer that skips Key/Value generation. It computes Queries and 
    cross-attends to a shared global KV cache generated by previous sequence blocks.
    """
    def __init__(self, dim, heads, lag, kv_heads=None, mlp_ratio=8/3, norm_eps=1e-6):
        super().__init__()
        self.heads = heads
        self.kv_heads = kv_heads if kv_heads is not None else heads
        self.head_dim = dim // heads
        self.lag = lag
        
        self.q_proj = nn.Linear(dim, self.heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        
        self.norm1 = RMSNorm(dim, eps=norm_eps)
        self.norm2 = RMSNorm(dim, eps=norm_eps)
        
        hidden_dim = int(mlp_ratio * dim)
        self.mlp = SwiGLU(dim, hidden_dim, dim)

    def forward(self, x, past_blossoms, start_idx, freqs_cis):
        B, L_q, C = x.shape
        
        # --- 1. Attention Block (Conditional) ---
        if start_idx >= self.lag and past_blossoms is not None:
            residual = x
            x_norm = self.norm1(x)
            
            _, L_k, _ = past_blossoms.shape
            
            q = self.q_proj(x_norm).view(B, L_q, self.heads, self.head_dim)
            q = apply_rotary_emb(q, freqs_cis[start_idx:start_idx + L_q])
            
            k, v = past_blossoms.chunk(2, dim=-1)
            k = k.view(B, L_k, self.kv_heads, self.head_dim)
            v = v.view(B, L_k, self.kv_heads, self.head_dim)
            
            # GQA Repeat
            if self.heads != self.kv_heads:
                n_rep = self.heads // self.kv_heads
                k = k.unsqueeze(3).expand(B, L_k, self.kv_heads, n_rep, self.head_dim).reshape(B, L_k, self.heads, self.head_dim)
                v = v.unsqueeze(3).expand(B, L_k, self.kv_heads, n_rep, self.head_dim).reshape(B, L_k, self.heads, self.head_dim)
            
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            
            q_pos = torch.arange(start_idx, start_idx + L_q, device=x.device).view(-1, 1)
            k_pos = torch.arange(0, L_k, device=x.device).view(1, -1)
            lag_mask = (k_pos <= (q_pos - self.lag)).unsqueeze(0).unsqueeze(0)
            
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=lag_mask)
            out = out.transpose(1, 2).reshape(B, L_q, C)
            x = residual + self.out_proj(out)
            
        # --- 2. MLP Block (Always Executed) ---
        x = x + self.mlp(self.norm2(x))
        return x


class BlossomLayer(nn.Module):
    """
    Executed after the final transformer layer for a sequence block. 
    It projects the final block representations into shared global Keys and 
    Values, applies RoPE, and writes them to the global blossom cache.
    """
    def __init__(self, dim, kv_heads, head_dim):
        super().__init__()
        self.kv_heads = kv_heads
        self.head_dim = head_dim
        self.k_proj = nn.Linear(dim, kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, kv_heads * head_dim, bias=False)

    def forward(self, x, start_idx, end_idx, blossom_cache, tracker, freqs_cis):
        B, L, _ = x.shape
        k = self.k_proj(x).view(B, L, self.kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, L, self.kv_heads, self.head_dim)
        
        # RoPE applied before broadcasting to the shared global cache
        k = apply_rotary_emb(k, freqs_cis[start_idx:end_idx])
        
        new_blossoms = torch.cat([k.flatten(2), v.flatten(2)], dim=-1)
        tracker = CacheWrite.apply(blossom_cache, start_idx, end_idx, new_blossoms, tracker)
        return x, tracker


# ---------------------------------------------------------
# 4. Model Orchestrator
# ---------------------------------------------------------
class BlossomModel(nn.Module):
    def __init__(self, vocab_size, dim, heads, kv_heads=None, num_layers=6, lag=32, block_size=32, max_seq_len=8192, mlp_ratio=8/3, norm_eps=1e-6):
        super().__init__()
        
        assert lag >= block_size, "Lag must be >= block_size so required blossoms are computed by previous sequence blocks."
        
        self.dim = dim
        self.heads = heads
        self.kv_heads = kv_heads if kv_heads is not None else heads
        self.head_dim = dim // heads
        self.block_size = block_size
        
        self.embed = nn.Embedding(vocab_size, dim)
        self.register_buffer("freqs_cis", precompute_freqs_cis(self.head_dim, max_seq_len), persistent=False)
        
        self.layers = nn.ModuleList()
        self.attention_layer_indices = []
        
        for i in range(num_layers):
            if i % 2 == 0:
                self.layers.append(AttentionLayer(dim, self.heads, self.kv_heads, mlp_ratio, norm_eps))
                self.attention_layer_indices.append(i)
            else:
                self.layers.append(CrossAttentionLayer(dim, self.heads, lag, self.kv_heads, mlp_ratio, norm_eps))
                
        self.norm_f = RMSNorm(dim, eps=norm_eps)
        
        # Layer to generate global KVs after the final transformer layer
        self.blossom_layer = BlossomLayer(dim, self.kv_heads, self.head_dim)
        self.classifier = nn.Linear(dim, vocab_size, bias=False)

    def process_block(self, x_emb, target, start_idx, end_idx, attention_caches, blossom_cache, tracker, freqs_cis):
        """Processes a sequence block through all layers, updating stateful caches."""
        x = x_emb
        
        past_blossoms = None
        if start_idx > 0:
            past_blossoms, tracker = CacheRead.apply(blossom_cache, start_idx, tracker)
        
        for i, layer in enumerate(self.layers):
            if isinstance(layer, AttentionLayer):
                cache_obj = attention_caches[i]
                x, tracker = layer(x, start_idx, end_idx, cache_obj, tracker, freqs_cis)
            else:
                x = layer(x, past_blossoms, start_idx, freqs_cis)
                
        x = self.norm_f(x)
        
        x, tracker = self.blossom_layer(x, start_idx, end_idx, blossom_cache, tracker, freqs_cis)
        
        logits = self.classifier(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1))
        
        return loss, tracker

    def forward(self, input_ids, targets):
        B, Total_Seq = input_ids.shape
        device = input_ids.device
        freqs_cis = self.freqs_cis.to(device)
        
        cache_dim = self.kv_heads * self.head_dim * 2
        
        attention_caches = {
            i: DifferentiableCache(B, Total_Seq, cache_dim, device)
            for i in self.attention_layer_indices
        }
        blossom_cache = DifferentiableCache(B, Total_Seq, cache_dim, device)
        
        tracker = torch.zeros(1, requires_grad=True, device=device)
        total_loss = 0.0
        
        for i in range(0, Total_Seq, self.block_size):
            end_idx = min(i + self.block_size, Total_Seq)
            x_block = self.embed(input_ids[:, i:end_idx])
            t_block = targets[:, i:end_idx]
            
            loss_block, tracker = checkpoint(
                self.process_block,
                x_block, t_block, i, end_idx, 
                attention_caches, blossom_cache, tracker, freqs_cis,
                use_reentrant=False
            )
            total_loss = total_loss + loss_block

        total_loss = total_loss + 0.0 * tracker.sum()
        
        num_blocks = math.ceil(Total_Seq / self.block_size)
        return total_loss / num_blocks
