# RUN ON: Colab/GPU
"""
distillation/student_architecture.py
======================================
Student transformer encoder built from scratch using only torch.nn.
This file is the architectural foundation of the FinCompress compression study.

DESIGN PHILOSOPHY:
  Every class and method in this file is intentionally written to be read as
  a teaching document. Comments explain the mathematical operations at each step
  of the attention mechanism — not just what is computed, but WHY it is computed
  that way and what the dimensions represent.

  The student is a miniaturized BERT-style encoder:
    - 4 layers  (teacher: 12)
    - 384 hidden dim (teacher: 768)
    - 6 attention heads (teacher: 12)
    - 1536 FFN intermediate size (teacher: 3072)

  This gives ~22M → ~4M parameter reduction while sharing the same tokenizer
  vocabulary as FinBERT (vocab_size=30522), enabling weight transfer during
  distillation.

Run:
    python -m fincompress.distillation.student_architecture
    (prints parameter summary and does a smoke-test forward pass)
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# HYPERPARAMETERS — all tunable values live here, never inline
# ============================================================
SEED = 42
NUM_CLASSES = 3
STUDENT_NUM_LAYERS = 4
STUDENT_HIDDEN_SIZE = 384
STUDENT_NUM_HEADS = 6
STUDENT_INTERMEDIATE_SIZE = 1536
STUDENT_DROPOUT = 0.1
VOCAB_SIZE = 30522          # Matches BERT/FinBERT tokenizer — required for
                             # embedding weight compatibility.
MAX_POSITION_EMBEDDINGS = 512
NUM_TOKEN_TYPES = 2          # BERT-style segment A / segment B


# ============================================================
# Attention module
# ============================================================

class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention as described in Vaswani et al. (2017).

    Splits the hidden dimension into num_heads parallel attention heads,
    each operating on a head_dim = hidden_size // num_heads subspace. The
    outputs are concatenated and projected back to hidden_size.

    Why multiple heads instead of one large attention?
    Each head can attend to different positions and capture different
    syntactic/semantic relationships simultaneously. A single head would
    be forced to average over all relationship types.

    Args:
        hidden_size: Dimensionality of input and output (d_model).
        num_heads: Number of parallel attention heads (must divide hidden_size).
        dropout: Dropout probability on attention weights.
    """

    def __init__(self, hidden_size: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        assert hidden_size % num_heads == 0, (
            f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})"
        )

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        # head_dim: each head operates on this many dimensions.
        # Splitting hidden_size evenly ensures heads have equal capacity.
        self.head_dim = hidden_size // num_heads

        # Q, K, V projections: each maps [batch, seq, hidden] → [batch, seq, hidden].
        # Using separate weights (rather than one 3×hidden projection) lets us
        # understand which part of the matrix encodes queries vs. keys vs. values.
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=True)

        # Output projection: merges concatenated head outputs back to hidden_size.
        # This learnable recombination allows the model to weight head contributions.
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=True)

        self.attn_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute multi-head self-attention.

        Args:
            x: Input tensor of shape [batch, seq_len, hidden_size].
            attention_mask: Optional mask of shape [batch, 1, 1, seq_len].
                Large negative values (e.g. -10000) suppress padding positions
                BEFORE softmax, driving their attention weights to ~0.

        Returns:
            Tuple of:
              - output: [batch, seq_len, hidden_size]  — attended representation
              - attn_weights: [batch, num_heads, seq_len, seq_len]  — for distillation
        """
        batch_size, seq_len, _ = x.shape

        # --- Q, K, V Projection ---
        # Each projection: [batch, seq, hidden] → [batch, seq, hidden]
        # The linear layers learn to produce query/key/value representations from
        # the same input sequence (this is SELF-attention).
        q = self.q_proj(x)  # what each position is looking for
        k = self.k_proj(x)  # what each position can offer as a match
        v = self.v_proj(x)  # the actual content to retrieve if matched

        # --- Reshape for multi-head computation ---
        # [batch, seq, hidden] → [batch, seq, num_heads, head_dim]
        #                       → [batch, num_heads, seq, head_dim]
        # Moving num_heads to dim 1 allows efficient batched matmul across heads.
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # --- Scaled dot-product attention ---
        # Step 1: Raw attention scores via Q·Kᵀ
        # [batch, num_heads, seq, head_dim] × [batch, num_heads, head_dim, seq]
        #   → [batch, num_heads, seq, seq]
        # Each (i, j) entry is the dot-product similarity between position i's query
        # and position j's key — how much position i "wants" to look at position j.
        attn_scores = torch.matmul(q, k.transpose(-2, -1))

        # Step 2: Scale by 1/√head_dim
        # Without this scaling, dot products grow large as head_dim increases,
        # pushing softmax into saturation regions with near-zero gradients.
        # Dividing by √head_dim keeps variance ≈ 1 regardless of head_dim.
        scale = math.sqrt(self.head_dim)
        attn_scores = attn_scores / scale

        # Step 3: Apply padding mask (if provided)
        # The mask adds a large negative value to padding positions so softmax
        # maps them to ~0, preventing the model from attending to padding tokens.
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask

        # Step 4: Softmax along the key dimension (last dim)
        # Converts raw scores to a probability distribution over positions.
        # Each query position produces a weighted combination of value vectors.
        attn_weights = F.softmax(attn_scores, dim=-1)

        # Step 5: Dropout on attention weights (regularization during training)
        # Randomly zeroing attention weights forces the model not to over-rely
        # on any single attention pattern — a form of structural regularization.
        attn_weights_dropped = self.attn_dropout(attn_weights)

        # Step 6: Weighted sum of value vectors
        # [batch, num_heads, seq, seq] × [batch, num_heads, seq, head_dim]
        #   → [batch, num_heads, seq, head_dim]
        # Each position's output is a weighted average of all value vectors,
        # weights determined by how much that position attends to each other.
        context = torch.matmul(attn_weights_dropped, v)

        # --- Concatenate heads and project ---
        # [batch, num_heads, seq, head_dim] → [batch, seq, num_heads, head_dim]
        #                                   → [batch, seq, hidden_size]
        # Concatenation reassembles all head outputs for the final projection.
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)

        # Output projection: learnable recombination of head outputs.
        # [batch, seq, hidden_size] → [batch, seq, hidden_size]
        output = self.out_proj(context)

        # Return both the output and the pre-dropout attention weights.
        # The pre-dropout weights are used for intermediate distillation —
        # we want to match the "intended" attention pattern, not the randomly
        # dropped one.
        return output, attn_weights


# ============================================================
# Transformer encoder layer
# ============================================================

class TransformerEncoderLayer(nn.Module):
    """
    One transformer encoder layer: self-attention → Add+Norm → FFN → Add+Norm.

    The residual connections (Add) are critical — without them, gradients
    vanish in deep networks. LayerNorm stabilizes activations after each
    sub-layer, enabling higher learning rates and faster convergence.

    The FFN (feed-forward network) acts as a per-position MLP: it first
    expands the representation to intermediate_size (4× expansion ratio,
    matching the BERT design principle) and then projects back. This is where
    most "knowledge storage" happens in transformer models.

    Args:
        hidden_size: Model dimension.
        num_heads: Number of attention heads.
        intermediate_size: Inner dimension of the FFN (typically 4× hidden_size).
        dropout: Dropout probability used throughout the layer.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        dropout: float,
    ) -> None:
        super().__init__()

        # Self-attention sub-layer
        self.attention = MultiHeadSelfAttention(hidden_size, num_heads, dropout)
        self.attn_norm = nn.LayerNorm(hidden_size)
        self.attn_dropout = nn.Dropout(dropout)

        # FFN sub-layer: expand → activate → contract
        # GELU (Gaussian Error Linear Unit) is preferred over ReLU in BERT-family
        # models — it's smooth and differentiable everywhere, which helps with
        # gradient flow in language model pre-training.
        self.ffn_fc1 = nn.Linear(hidden_size, intermediate_size)
        self.ffn_act = nn.GELU()
        self.ffn_dropout = nn.Dropout(dropout)
        self.ffn_fc2 = nn.Linear(intermediate_size, hidden_size)
        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn_out_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply one encoder layer.

        Args:
            x: [batch, seq_len, hidden_size]
            attention_mask: Optional padding mask.

        Returns:
            Tuple of (output [batch, seq, hidden], attn_weights [batch, heads, seq, seq]).
        """
        # --- Attention sub-layer with pre-norm residual ---
        attn_out, attn_weights = self.attention(x, attention_mask)
        # Residual: x + attention_output keeps gradient paths short.
        x = self.attn_norm(x + self.attn_dropout(attn_out))

        # --- FFN sub-layer with pre-norm residual ---
        ffn_out = self.ffn_fc2(
            self.ffn_dropout(
                self.ffn_act(
                    self.ffn_fc1(x)
                )
            )
        )
        x = self.ffn_norm(x + self.ffn_out_dropout(ffn_out))

        return x, attn_weights


# ============================================================
# Transformer encoder (stack of layers)
# ============================================================

class TransformerEncoder(nn.Module):
    """
    A stack of TransformerEncoderLayer modules.

    Returns hidden states AND attention weights from EVERY layer so that
    intermediate distillation can supervise any subset of layers.

    Args:
        num_layers: How many encoder layers to stack.
        hidden_size: Model dimension.
        num_heads: Attention heads per layer.
        intermediate_size: FFN inner dimension.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        num_layers: int,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(hidden_size, num_heads, intermediate_size, dropout)
            for _ in range(num_layers)
        ])

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """
        Pass input through all encoder layers and collect outputs.

        Args:
            x: [batch, seq_len, hidden_size] — typically embedding output.
            attention_mask: Optional padding mask.

        Returns:
            Tuple of:
              - all_hidden_states: list of [batch, seq, hidden] — one per layer
              - all_attn_weights: list of [batch, heads, seq, seq] — one per layer
        """
        all_hidden_states: list[torch.Tensor] = []
        all_attn_weights: list[torch.Tensor] = []

        current = x
        for layer in self.layers:
            current, attn_weights = layer(current, attention_mask)
            all_hidden_states.append(current)
            all_attn_weights.append(attn_weights)

        return all_hidden_states, all_attn_weights


# ============================================================
# Full student classifier
# ============================================================

class StudentClassifier(nn.Module):
    """
    Complete student model: embeddings → transformer encoder → classifier head.

    Architecture mirrors BERT exactly (same embedding scheme, same classification
    head design) so that the teacher's tokenizer outputs map directly to student
    inputs without any preprocessing changes. This is intentional — it lets us
    use the exact same DataLoader for teacher and student during distillation.

    The forward() method returns a dict so callers can selectively access logits,
    hidden states, or attention weights without always computing all three.

    Args:
        vocab_size: Vocabulary size (30522 for BERT/FinBERT tokenizer).
        hidden_size: Embedding and encoder hidden dimension.
        num_layers: Number of encoder layers.
        num_heads: Attention heads per layer.
        intermediate_size: FFN inner dimension.
        dropout: Dropout probability throughout.
        num_classes: Number of classification labels.
    """

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        hidden_size: int = STUDENT_HIDDEN_SIZE,
        num_layers: int = STUDENT_NUM_LAYERS,
        num_heads: int = STUDENT_NUM_HEADS,
        intermediate_size: int = STUDENT_INTERMEDIATE_SIZE,
        dropout: float = STUDENT_DROPOUT,
        num_classes: int = NUM_CLASSES,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads

        # --- Embeddings (BERT-style) ---
        # Token embedding: maps token IDs to dense vectors.
        self.token_embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)

        # Position embedding: adds positional information.
        # BERT uses learned (not sinusoidal) positional embeddings because
        # domain fine-tuning benefits from position representations that adapt.
        self.position_embedding = nn.Embedding(MAX_POSITION_EMBEDDINGS, hidden_size)

        # Segment embedding: distinguishes sentence A vs. B in pair tasks.
        # Even for single-sentence tasks we always pass all-zeros segment IDs,
        # so this adds a constant offset — effectively a learned bias per segment.
        self.segment_embedding = nn.Embedding(NUM_TOKEN_TYPES, hidden_size)

        self.embedding_norm = nn.LayerNorm(hidden_size)
        self.embedding_dropout = nn.Dropout(dropout)

        # --- Transformer encoder ---
        self.encoder = TransformerEncoder(
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_heads=num_heads,
            intermediate_size=intermediate_size,
            dropout=dropout,
        )

        # --- Classification head ---
        # Uses only the [CLS] token's representation (index 0), following BERT
        # convention. The [CLS] token is trained to aggregate sentence-level info.
        self.classifier = nn.Linear(hidden_size, num_classes)

        # Initialize weights (mirrors HuggingFace BERT initialization)
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with truncated normal (σ=0.02), matching BERT."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _build_attention_mask(self, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Convert a binary attention mask [batch, seq] to an additive mask
        [batch, 1, 1, seq] compatible with the attention score tensor.

        Positions where attention_mask == 0 (padding) receive -10000 so that
        softmax maps them to ~0 attention weight.

        Args:
            attention_mask: Binary tensor [batch, seq_len], 1=real token, 0=padding.

        Returns:
            Additive mask [batch, 1, 1, seq_len] ready for addition to scores.
        """
        # Flip: 0 (padding) → large negative, 1 (real) → 0 (no effect)
        extended = (1.0 - attention_mask.float()) * -10000.0
        return extended.unsqueeze(1).unsqueeze(2)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        """
        Full forward pass: embeddings → encoder → classifier.

        Args:
            input_ids: Token ID tensor [batch, seq_len].
            attention_mask: Binary mask [batch, seq_len]. Defaults to all-ones.
            token_type_ids: Segment IDs [batch, seq_len]. Defaults to all-zeros.

        Returns:
            Dict with keys:
              "logits"           — [batch, num_classes] classification scores
              "hidden_states"    — list of [batch, seq, hidden], one per layer
              "attention_weights"— list of [batch, heads, seq, seq], one per layer
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Default masks if not provided
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, seq_len, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)

        # Position IDs: [0, 1, 2, ..., seq_len-1] expanded to [batch, seq_len]
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)

        # --- Embedding layer ---
        # Summing all three embedding types is BERT's design: each contributes
        # complementary information (what, where, which-sentence).
        embeddings = (
            self.token_embedding(input_ids)
            + self.position_embedding(position_ids)
            + self.segment_embedding(token_type_ids)
        )
        embeddings = self.embedding_dropout(self.embedding_norm(embeddings))

        # --- Encoder ---
        additive_mask = self._build_attention_mask(attention_mask)
        all_hidden_states, all_attn_weights = self.encoder(embeddings, additive_mask)

        # --- Classification ---
        # [CLS] representation: the first token aggregates sequence-level context.
        cls_repr = all_hidden_states[-1][:, 0, :]  # [batch, hidden]
        logits = self.classifier(cls_repr)           # [batch, num_classes]

        return {
            "logits": logits,
            "hidden_states": all_hidden_states,
            "attention_weights": all_attn_weights,
        }

    def count_parameters(self) -> int:
        """
        Print a per-component parameter breakdown and return total count.

        Returns:
            Total number of trainable parameters.
        """
        components = {
            "token_embedding": self.token_embedding,
            "position_embedding": self.position_embedding,
            "segment_embedding": self.segment_embedding,
            "embedding_norm": self.embedding_norm,
            "encoder": self.encoder,
            "classifier": self.classifier,
        }

        total = 0
        print(f"\n{'Component':<22} {'Parameters':>12}")
        print("-" * 36)
        for name, module in components.items():
            count = sum(p.numel() for p in module.parameters() if p.requires_grad)
            total += count
            print(f"{name:<22} {count:>12,}")
        print("-" * 36)
        print(f"{'TOTAL':<22} {total:>12,}")

        return total


# ============================================================
# Smoke test
# ============================================================

def main() -> None:
    """Smoke-test: instantiate student, run a forward pass, print param count."""
    torch.manual_seed(SEED)

    model = StudentClassifier()
    model.eval()

    # Fake batch: batch_size=2, seq_len=16
    batch_size, seq_len = 2, 16
    input_ids = torch.randint(0, VOCAB_SIZE, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    attention_mask[0, 12:] = 0  # simulate padding in sample 0

    with torch.no_grad():
        out = model(input_ids, attention_mask)

    print("StudentClassifier smoke test")
    print(f"  logits shape:           {out['logits'].shape}")
    print(f"  hidden_states[0] shape: {out['hidden_states'][0].shape}")
    print(f"  attn_weights[0] shape:  {out['attention_weights'][0].shape}")
    print(f"  num hidden layers:      {len(out['hidden_states'])}")
    print(f"  num attn layers:        {len(out['attention_weights'])}")

    model.count_parameters()


if __name__ == "__main__":
    main()
