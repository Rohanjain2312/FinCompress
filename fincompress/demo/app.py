# RUN ON: Local/CPU
"""
demo/app.py
============
Gradio web app for the FinCompress demo — compare all 7 compressed model
variants on financial sentiment in real time.

The purpose of this demo is to make the compression trade-offs visceral:
you can feel the latency difference, not just read about it in a table.

Run:
    python -m fincompress.demo.app
"""

import time
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
import gradio as gr
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from fincompress.distillation.student_architecture import (
    StudentClassifier,
    STUDENT_NUM_LAYERS,
    STUDENT_HIDDEN_SIZE,
    STUDENT_NUM_HEADS,
    STUDENT_INTERMEDIATE_SIZE,
    STUDENT_DROPOUT,
)

# ============================================================
# HYPERPARAMETERS — all tunable values live here, never inline
# ============================================================
SEED = 42
MAX_SEQ_LEN = 128
NUM_CLASSES = 3
LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}

# Number of inference runs to median-average for displayed latency
DEMO_LATENCY_RUNS = 3

# Color codes for prediction cells
LABEL_COLORS = {
    "positive": "#d4edda",
    "negative": "#f8d7da",
    "neutral": "#e2e3e5",
}

# ============================================================
# Paths
# ============================================================
PROJECT_ROOT = Path(__file__).parent.parent
CHECKPOINTS_ROOT = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"
TEACHER_TOKENIZER_DIR = CHECKPOINTS_ROOT / "teacher" / "tokenizer"

# ============================================================
# Model registry
# ============================================================

MODEL_SPECS: list[dict] = [
    {
        "name": "teacher",
        "label": "Teacher (FinBERT)",
        "path": CHECKPOINTS_ROOT / "teacher",
        "is_teacher": True,
        "is_quantized": False,
    },
    {
        "name": "student_vanilla_kd",
        "label": "Vanilla KD",
        "path": CHECKPOINTS_ROOT / "student_vanilla_kd",
        "is_teacher": False,
        "is_quantized": False,
    },
    {
        "name": "student_intermediate_kd",
        "label": "Intermediate KD",
        "path": CHECKPOINTS_ROOT / "student_intermediate_kd",
        "is_teacher": False,
        "is_quantized": False,
    },
    {
        "name": "student_ptq",
        "label": "INT8 PTQ",
        "path": CHECKPOINTS_ROOT / "student_ptq",
        "is_teacher": False,
        "is_quantized": True,
    },
    {
        "name": "student_qat",
        "label": "INT8 QAT",
        "path": CHECKPOINTS_ROOT / "student_qat",
        "is_teacher": False,
        "is_quantized": True,
    },
    {
        "name": "pruned_teacher_30pct",
        "label": "Pruned 30%",
        "path": CHECKPOINTS_ROOT / "pruned_teacher_30pct",
        "is_teacher": True,
        "is_quantized": False,
    },
    {
        "name": "pruned_teacher_50pct",
        "label": "Pruned 50%",
        "path": CHECKPOINTS_ROOT / "pruned_teacher_50pct",
        "is_teacher": True,
        "is_quantized": False,
    },
]


# ============================================================
# Model loading
# ============================================================

def try_load_tokenizer():
    """Attempt to load the shared tokenizer, return None on failure."""
    if TEACHER_TOKENIZER_DIR.exists():
        try:
            return AutoTokenizer.from_pretrained(str(TEACHER_TOKENIZER_DIR))
        except Exception:
            return None
    return None


def try_load_model(spec: dict) -> Optional[torch.nn.Module]:
    """
    Attempt to load a model from its checkpoint. Returns None gracefully
    on any error so the demo degrades rather than crashes.

    Args:
        spec: Model specification dict from MODEL_SPECS.

    Returns:
        Loaded model in eval mode, or None if checkpoint not available.
    """
    path: Path = spec["path"]
    if not path.exists():
        return None

    try:
        if spec["is_teacher"]:
            model = AutoModelForSequenceClassification.from_pretrained(str(path))
        else:
            model = StudentClassifier(
                hidden_size=STUDENT_HIDDEN_SIZE,
                num_layers=STUDENT_NUM_LAYERS,
                num_heads=STUDENT_NUM_HEADS,
                intermediate_size=STUDENT_INTERMEDIATE_SIZE,
                dropout=STUDENT_DROPOUT,
                num_classes=NUM_CLASSES,
            )
            if spec["is_quantized"]:
                torch.backends.quantized.engine = "fbgemm"
                model.qconfig = torch.quantization.get_default_qconfig("fbgemm")
                model.token_embedding.qconfig = None
                model.position_embedding.qconfig = None
                model.segment_embedding.qconfig = None
                model.classifier.qconfig = None
                torch.quantization.prepare(model, inplace=True)
                torch.quantization.convert(model, inplace=True)
            state_dict = torch.load(path / "pytorch_model.bin", map_location="cpu")
            model.load_state_dict(state_dict, strict=False)

        model.eval()
        return model
    except Exception:
        return None


def get_model_size_mb(path: Path, spec: dict) -> Optional[float]:
    """
    Read size from checkpoint_info.json, or compute from model weights.

    Args:
        path: Checkpoint directory.
        spec: Model spec dict.

    Returns:
        Size in MB, or None if unavailable.
    """
    info_path = path / "checkpoint_info.json"
    if info_path.exists():
        import json
        try:
            with open(info_path) as f:
                return json.load(f).get("size_mb")
        except Exception:
            pass
    return None


# ============================================================
# Inference helper
# ============================================================

def run_inference(
    model: torch.nn.Module,
    tokenizer,
    text: str,
    is_teacher: bool,
) -> tuple[str, float, float]:
    """
    Tokenize text, run DEMO_LATENCY_RUNS forward passes, return prediction.

    Args:
        model: Loaded model in eval mode.
        tokenizer: Shared tokenizer.
        text: Input financial sentence.
        is_teacher: Whether model uses HuggingFace or StudentClassifier API.

    Returns:
        Tuple of (label_str, confidence_pct, median_latency_ms).
    """
    enc = tokenizer(
        text,
        max_length=MAX_SEQ_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    token_type_ids = enc.get("token_type_ids", torch.zeros_like(input_ids))

    latencies: list[float] = []
    logits_out = None

    with torch.no_grad():
        for i in range(DEMO_LATENCY_RUNS):
            t0 = time.perf_counter()
            if is_teacher:
                out = model(input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids)
                logits_out = out.logits
            else:
                out = model(input_ids, attention_mask, token_type_ids)
                logits_out = out["logits"]
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)

    # Sort and take the middle value as median
    latencies_sorted = sorted(latencies)
    median_ms = latencies_sorted[len(latencies_sorted) // 2]

    probs = torch.softmax(logits_out, dim=-1).squeeze(0)
    predicted_idx = probs.argmax().item()
    label = LABEL_MAP[predicted_idx]
    confidence = float(probs[predicted_idx]) * 100.0

    return label, confidence, median_ms


# ============================================================
# App setup
# ============================================================

# Load everything at startup so UI is responsive during inference
print("FinCompress Demo — loading models...")
TOKENIZER = try_load_tokenizer()

LOADED_MODELS: list[dict] = []
for spec in MODEL_SPECS:
    model = try_load_model(spec)
    size_mb = get_model_size_mb(spec["path"], spec) if spec["path"].exists() else None
    LOADED_MODELS.append({
        **spec,
        "model": model,
        "size_mb": size_mb,
        "available": model is not None,
    })
    status = "loaded" if model is not None else "not found"
    print(f"  {spec['label']:<25} {status}")

# Load benchmark F1 scores if available
BENCHMARK_F1: dict[str, Optional[float]] = {spec["name"]: None for spec in MODEL_SPECS}
bench_csv = RESULTS_DIR / "benchmark_results.csv"
if bench_csv.exists():
    try:
        df_bench = pd.read_csv(bench_csv)
        for _, row in df_bench.iterrows():
            model_name = row.get("model_name")
            if model_name in BENCHMARK_F1 and "macro_f1" in row:
                BENCHMARK_F1[model_name] = round(float(row["macro_f1"]), 4)
    except Exception:
        pass

print("Startup complete.\n")


# ============================================================
# Inference callback
# ============================================================

def run_all_models(text: str) -> str:
    """
    Run inference on all loaded models and return an HTML comparison table.

    Args:
        text: Input financial sentence.

    Returns:
        HTML string with the results table.
    """
    if not text.strip():
        return "<p style='color:grey'>Please enter a financial sentence above.</p>"

    if TOKENIZER is None:
        return "<p style='color:red'>Tokenizer not found. Run teacher training first.</p>"

    rows_html = ""
    for entry in LOADED_MODELS:
        label = entry["label"]
        size_mb_str = f"{entry['size_mb']:.1f}" if entry["size_mb"] is not None else "—"
        f1_str = f"{BENCHMARK_F1[entry['name']]:.4f}" if BENCHMARK_F1.get(entry["name"]) else "—"

        if not entry["available"]:
            rows_html += (
                f"<tr>"
                f"<td><b>{label}</b></td>"
                f"<td>—</td><td>—</td><td>—</td>"
                f"<td>{size_mb_str}</td><td>{f1_str}</td>"
                f"</tr>"
            )
            continue

        try:
            pred_label, confidence, latency_ms = run_inference(
                entry["model"], TOKENIZER, text, entry["is_teacher"]
            )
            color = LABEL_COLORS.get(pred_label, "#ffffff")
            rows_html += (
                f"<tr>"
                f"<td><b>{label}</b></td>"
                f"<td style='background:{color};text-align:center'>{pred_label}</td>"
                f"<td style='text-align:center'>{confidence:.1f}%</td>"
                f"<td style='text-align:center'>{latency_ms:.1f} ms</td>"
                f"<td style='text-align:center'>{size_mb_str}</td>"
                f"<td style='text-align:center'>{f1_str}</td>"
                f"</tr>"
            )
        except Exception as e:
            rows_html += (
                f"<tr>"
                f"<td><b>{label}</b></td>"
                f"<td colspan='5' style='color:red'>Error: {e}</td>"
                f"</tr>"
            )

    html = f"""
    <table style='width:100%;border-collapse:collapse;font-family:monospace;font-size:14px'>
      <thead>
        <tr style='background:#f0f0f0'>
          <th style='text-align:left;padding:8px'>Model</th>
          <th style='padding:8px'>Prediction</th>
          <th style='padding:8px'>Confidence %</th>
          <th style='padding:8px'>Latency (ms)</th>
          <th style='padding:8px'>Size (MB)</th>
          <th style='padding:8px'>F1 (test)</th>
        </tr>
      </thead>
      <tbody>
        {rows_html}
      </tbody>
    </table>
    <p style='font-style:italic;font-size:12px;color:grey;margin-top:8px'>
      Latency measured on CPU (Apple Silicon / x86). F1 from held-out test set.
    </p>
    """
    return html


# ============================================================
# Gradio interface
# ============================================================

with gr.Blocks(title="FinCompress — FinBERT Compression Benchmark") as demo:
    gr.Markdown(
        "# FinCompress — FinBERT Compression Benchmark\n"
        "**Compare 7 compressed model variants on financial sentiment in real time**\n\n"
        "Each row shows one trained model variant. "
        "Latency is measured live on your machine."
    )

    with gr.Row():
        text_input = gr.Textbox(
            label="Financial Sentence",
            placeholder=(
                "Enter a financial headline "
                "(e.g. 'Apple reports record quarterly earnings')"
            ),
            lines=3,
        )

    run_btn = gr.Button("Run Inference", variant="primary")
    output_html = gr.HTML(
        value="<p style='color:grey'>Results will appear here after you click Run Inference.</p>"
    )

    run_btn.click(fn=run_all_models, inputs=text_input, outputs=output_html)
    text_input.submit(fn=run_all_models, inputs=text_input, outputs=output_html)

    gr.Markdown(
        "---\n"
        "**Color coding:** "
        "<span style='background:#d4edda;padding:2px 6px'>positive</span> &nbsp;"
        "<span style='background:#f8d7da;padding:2px 6px'>negative</span> &nbsp;"
        "<span style='background:#e2e3e5;padding:2px 6px'>neutral</span>"
    )


def main() -> None:
    """Launch the Gradio demo server."""
    demo.launch(share=False)


if __name__ == "__main__":
    main()
