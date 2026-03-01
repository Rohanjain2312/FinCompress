"""
FinCompress — HuggingFace Space
================================
Gradio 6 demo showcasing FinBERT compression via knowledge distillation,
INT8 quantization, and structured pruning.

Deployment fixes applied (from deployment-issues.md):
  #2  short_description ≤ 60 chars            → in README.md frontmatter
  #3  gradio==6.6.0                            → in requirements.txt
  #4  dtype= not torch_dtype=                 → pipeline() calls below
  #5  blocking pre-warm before launch()       → models loaded at module level
  #6  asyncio.get_running_loop() not get_event_loop() → no asyncio used at all
  #7  theme/css in gr.Blocks(), not launch()  → see Blocks() call below
  #8  server_name="0.0.0.0" + PORT env var   → demo.launch() call below
  #9/#10 Gradio 6 queue (safe_get_lock fix)  → gradio==6.6.0
"""

import os
import time

import torch
import gradio as gr
from transformers import AutoTokenizer, pipeline
from huggingface_hub import hf_hub_download

# Local copy of our custom student architecture (no package install needed)
from student_architecture import (
    StudentClassifier,
    STUDENT_NUM_LAYERS,
    STUDENT_HIDDEN_SIZE,
    STUDENT_NUM_HEADS,
    STUDENT_INTERMEDIATE_SIZE,
    STUDENT_DROPOUT,
)

# ── Constants ──────────────────────────────────────────────────────────────────
TEACHER_MODEL_ID  = "ProsusAI/finbert"
STUDENT_REPO_ID   = "rohanjain2312/FinCompress_student"
STUDENT_FILENAME  = "pytorch_model.bin"
MAX_SEQ_LEN       = 128
NUM_CLASSES       = 3

# FinBERT outputs lowercase labels; our student uses 0=neg,1=neu,2=pos
TEACHER_LABEL_MAP = {
    "positive": "Positive 📈",
    "negative": "Negative 📉",
    "neutral":  "Neutral  ➖",
}
STUDENT_LABEL_MAP = {
    0: "Negative 📉",
    1: "Neutral  ➖",
    2: "Positive 📈",
}

# ── Baked-in benchmark results (from checkpoint_info.json after Colab run) ────
BENCHMARK_ROWS = [
    ["Teacher (FinBERT)",         "Fine-tuning",          "109M", "437.9", "0.8876", "baseline"],
    ["Student — Vanilla KD",      "Soft-label KD",         "19M",  "76.1", "0.8017", "5.8× smaller"],
    ["Student — Intermediate KD", "Hidden + Attn KD",      "19M",  "76.1", "0.7712", "5.8× smaller"],
    ["Student — PTQ (INT8)",      "Post-Training Quant",   "12M",  "47.7", "0.7712", "9.1× smaller"],
    ["Student — QAT (INT8)",      "Quant-Aware Training",  "12M",  "47.7", "0.7601", "9.1× smaller"],
    ["Pruned Teacher 30%",        "Structured Pruning",   "109M", "437.9", "0.8966", "↑ beats teacher!"],
    ["Pruned Teacher 50%",        "Structured Pruning",   "109M", "437.9", "0.8936", "↑ beats teacher!"],
]
BENCHMARK_COLS = ["Model", "Technique", "Params", "Size (MB)", "Val Macro F1", "vs Teacher"]

EXAMPLES = [
    ["The company reported record profits, beating analyst expectations by 20%."],
    ["Inflation continues to rise as the Federal Reserve maintains its current policy."],
    ["The startup filed for bankruptcy after failing to secure Series B funding."],
    ["Oil prices remain stable amid ongoing geopolitical tensions in the Middle East."],
    ["Tech stocks surged following strong earnings reports across the sector."],
    ["The merger was called off due to regulatory concerns from the antitrust division."],
]

# ── Model loading — BLOCKING before launch() (fix #5: no daemon threads) ──────
print("── [1/3] Loading teacher (ProsusAI/finbert)…")
teacher_pipe = pipeline(
    "text-classification",
    model=TEACHER_MODEL_ID,
    return_all_scores=True,   # return probability for every class
    device="cpu",             # HF free tier is CPU-only
    # NOTE: dtype= (not torch_dtype=) is the correct kwarg since transformers 4.40
    # For text-classification we don't pass dtype — defaults to float32 which is correct.
)
print("✅ Teacher ready.")

print("── [2/3] Loading tokenizer…")
tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL_ID)
print("✅ Tokenizer ready.")

print("── [3/3] Loading student from HuggingFace Hub…")
STUDENT_LOADED = False
student = None
try:
    model_path = hf_hub_download(repo_id=STUDENT_REPO_ID, filename=STUDENT_FILENAME)
    student = StudentClassifier(
        hidden_size=STUDENT_HIDDEN_SIZE,
        num_layers=STUDENT_NUM_LAYERS,
        num_heads=STUDENT_NUM_HEADS,
        intermediate_size=STUDENT_INTERMEDIATE_SIZE,
        dropout=STUDENT_DROPOUT,
        num_classes=NUM_CLASSES,
    )
    state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
    student.load_state_dict(state_dict, strict=False)
    student.eval()
    STUDENT_LOADED = True
    n_params = sum(p.numel() for p in student.parameters())
    print(f"✅ Student ready ({n_params:,} params).")
except Exception as exc:
    print(f"⚠️  Student not loaded: {exc}")
    print("   Upload pytorch_model.bin to rohanjain2312/FinCompress_student to enable it.")


# ── Inference function ─────────────────────────────────────────────────────────
def analyze(text: str):
    """Run teacher + student inference and return confidence dicts + latencies."""
    if not text or not text.strip():
        empty = {"Positive 📈": 0.0, "Neutral  ➖": 0.0, "Negative 📉": 0.0}
        return empty, "—", empty, "—", ""

    text = text.strip()

    # ── Teacher ───────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    raw = teacher_pipe(text)
    teacher_ms = (time.perf_counter() - t0) * 1000

    # pipeline returns list-of-dicts for single input when return_all_scores=True
    teacher_results = raw[0] if isinstance(raw[0], list) else raw
    teacher_probs = {
        TEACHER_LABEL_MAP[r["label"]]: round(r["score"], 4)
        for r in teacher_results
    }

    # ── Student ───────────────────────────────────────────────────────────────
    if STUDENT_LOADED:
        enc = tokenizer(
            text,
            max_length=MAX_SEQ_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids      = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        token_type_ids = enc.get("token_type_ids", torch.zeros_like(input_ids))

        t0 = time.perf_counter()
        with torch.no_grad():
            out = student(input_ids, attention_mask, token_type_ids)
        student_ms = (time.perf_counter() - t0) * 1000

        probs = torch.softmax(out["logits"][0], dim=-1)
        student_probs = {
            STUDENT_LABEL_MAP[i]: round(float(probs[i]), 4)
            for i in range(NUM_CLASSES)
        }
        speedup    = teacher_ms / max(student_ms, 0.1)
        comparison = (
            f"⚡ Student is **{speedup:.1f}×** faster on this sentence  "
            f"({student_ms:.0f} ms vs {teacher_ms:.0f} ms teacher)  |  "
            f"5.8× smaller model  |  −8.6 F1 pts on val set"
        )
    else:
        student_probs = {"Positive 📈": 0.0, "Neutral  ➖": 0.0, "Negative 📉": 0.0}
        student_ms    = 0.0
        comparison    = "⚠️ Student weights not uploaded yet — see model repo."

    return (
        teacher_probs,
        f"⏱️ {teacher_ms:.0f} ms",
        student_probs,
        f"⏱️ {student_ms:.0f} ms" if STUDENT_LOADED else "—",
        comparison,
    )


# ── Gradio 6 UI ───────────────────────────────────────────────────────────────
# Fix #7: theme= and css= belong in gr.Blocks(), NOT in launch()
css = """
.speed-banner {
    text-align: center;
    font-size: 1.05em;
    padding: 10px 16px;
    border-radius: 8px;
    background: #e8f5e9;
    margin-top: 8px;
}
.teacher-col { border-right: 2px solid #e0e0e0; padding-right: 16px; }
footer { display: none !important; }
"""

with gr.Blocks(
    title="FinCompress — Financial Sentiment Compression",
    theme=gr.themes.Soft(),   # fix #7: theme in Blocks, not launch()
    css=css,
) as demo:

    gr.Markdown(
        """
        # 🗜️ FinCompress — Financial Sentiment Compression
        Compressing **FinBERT** (109M params, 438 MB) into a **19M-param student** (76 MB)
        using knowledge distillation — then pushing further with INT8 quantization (48 MB) and
        structured attention-head pruning. All trained and benchmarked on financial sentiment.

        **Teacher → Student: 5.8× smaller · 9.1× smaller with INT8 quantization**
        """
    )

    with gr.Tabs():

        # ── Tab 1: Live Demo ──────────────────────────────────────────────────
        with gr.Tab("🔍 Live Demo"):
            gr.Markdown(
                "_Type any financial sentence below and click **Analyze** to compare "
                "the teacher (FinBERT, 109M) and student (Vanilla KD, 19M) side-by-side._"
            )

            with gr.Row():
                text_input = gr.Textbox(
                    label="Financial sentence",
                    placeholder="e.g. The company reported record profits, beating analyst expectations…",
                    lines=3,
                    scale=4,
                )
                analyze_btn = gr.Button("Analyze Sentiment ▶", variant="primary", scale=1)

            with gr.Row():
                with gr.Column(elem_classes=["teacher-col"]):
                    gr.Markdown("### 🎓 Teacher — FinBERT\n`109M params · 438 MB · FP32`")
                    teacher_label   = gr.Label(num_top_classes=3, label="Confidence scores")
                    teacher_latency = gr.Textbox(label="Inference time", interactive=False)

                with gr.Column():
                    gr.Markdown("### 🧑‍🎓 Student — Vanilla KD\n`19M params · 76 MB · FP32 · 5.8× smaller`")
                    student_label   = gr.Label(num_top_classes=3, label="Confidence scores")
                    student_latency = gr.Textbox(label="Inference time", interactive=False)

            speed_banner = gr.Markdown("", elem_classes=["speed-banner"])

            gr.Examples(
                examples=EXAMPLES,
                inputs=text_input,
                label="Example sentences — click to load",
            )

            outputs = [teacher_label, teacher_latency, student_label, student_latency, speed_banner]
            analyze_btn.click(fn=analyze, inputs=text_input, outputs=outputs)
            text_input.submit(fn=analyze,  inputs=text_input, outputs=outputs)

        # ── Tab 2: Benchmark Results ──────────────────────────────────────────
        with gr.Tab("📊 Benchmark Results"):
            gr.Markdown(
                """
                ### All 7 model variants — held-out test set, CPU inference

                | Highlight | Result |
                |---|---|
                | **Best compression** | PTQ / QAT student — **47.7 MB** (9.1× smaller than teacher) |
                | **Best accuracy-size tradeoff** | Vanilla KD — **0.8017 F1** at 76 MB |
                | **Surprising finding** | Pruning 30–50% of attention heads *improves* F1 (+0.9 pts) |
                | **Why pruning helps** | Removing redundant heads reduces overfitting — a regularization effect |
                """
            )

            gr.DataFrame(
                value=BENCHMARK_ROWS,
                headers=BENCHMARK_COLS,
                label="Full benchmark",
                interactive=False,
                wrap=True,
            )

            gr.Markdown(
                """
                > **Metrics**: Val Macro F1 on `financial_phrasebank` (sentences_allagree split).
                > Latency measured as median over 500 single-sample CPU runs with 50 warmup iterations.
                > Training hardware: Google Colab T4 GPU. Benchmarking hardware: CPU.
                """
            )

        # ── Tab 3: Architecture & Methods ────────────────────────────────────
        with gr.Tab("🏗️ Architecture & Methods"):
            gr.Markdown(
                """
                ## FinCompress Compression Pipeline

                **Starting point:** ProsusAI/finbert — BERT-base further pre-trained on 4.9B
                tokens of financial text, then fine-tuned on `financial_phrasebank`.
                Result: **109M params, 438 MB, 0.888 val Macro F1**.

                ---

                ### 1 · Knowledge Distillation

                Train a **4-layer, 384-hidden, 6-head student** (19M params) to mimic the teacher.

                **Vanilla KD** — soft-label loss:
                ```
                L = α · T² · KL(student_soft ‖ teacher_soft) + (1−α) · CE(student, hard_labels)
                ```
                Temperature T=4 softens the teacher's distribution so the student learns
                uncertainty structure, not just the argmax label.

                **Intermediate KD** — adds layer-to-layer supervision:
                ```
                L += λ₁ · MSE(proj(student_hidden_i), teacher_hidden_j)
                   + λ₂ · MSE(student_attn_i, teacher_attn_j)
                ```
                Layer mapping: `{0→2, 1→5, 2→8, 3→11}` — evenly spaced across the 12-layer teacher.

                **Result:** 5.8× smaller, 0.802 vs 0.888 F1 (−8.6 pts).

                ---

                ### 2 · INT8 Quantization

                Reduces FP32 weights to INT8, cutting the model to **47.7 MB (9.1× smaller)**.

                - **PTQ** (Post-Training Quantization): `torch.quantization.quantize_dynamic` on the
                  pre-trained FP32 student — zero extra training. F1 unchanged (0.771).
                - **QAT** (Quantization-Aware Training): fine-tune with fake-quant + straight-through
                  estimator so weights adapt to INT8 noise. Slight F1 dip (0.760) here, but
                  typically more robust on unseen domains.

                ---

                ### 3 · Structured Attention-Head Pruning

                Remove entire attention heads from the **teacher** using entropy-based importance:

                1. Compute attention entropy per head over the validation set
                2. Low-entropy heads (near-uniform distributions) carry little information — prune them
                3. Fine-tune for 3 epochs to recover; repeat up to 5 rounds

                **Surprising result:** Removing 30–50% of heads *improves* val F1 by +0.9 pts.
                Redundant heads act as noise — pruning them regularises the model.

                ---

                ### Student Architecture
                ```
                StudentClassifier
                ├── token_embedding    [30 522 × 384]
                ├── position_embedding [  512  × 384]
                ├── segment_embedding  [    2  × 384]
                ├── TransformerEncoder  (4 layers)
                │   └── MultiHeadSelfAttention (6 heads, head_dim = 64)
                │       └── FFN  384 → 1 536 → 384  (GELU activation)
                └── classifier          [384 → 3]
                Total: 19 017 603 parameters
                ```

                ---

                ### Links
                - 📦 [GitHub — FinCompress](https://github.com/Rohanjain2312/FinCompress)
                - 🤗 [Student Model Weights](https://huggingface.co/rohanjain2312/FinCompress_student)
                - 📊 [Dataset: financial_phrasebank](https://huggingface.co/datasets/takala/financial_phrasebank)
                """
            )

# ── Launch ────────────────────────────────────────────────────────────────────
# Fix #8: bind to 0.0.0.0 so the HF Spaces reverse proxy can reach the server.
#         Read PORT from environment (HF Spaces injects it at runtime).
# Fix #7: no theme= or css= here — they live in gr.Blocks() above.
demo.launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 7860)),
)
