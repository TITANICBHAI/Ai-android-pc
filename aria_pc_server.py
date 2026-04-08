#!/usr/bin/env python3
"""
aria_pc_server.py — ARIA PC Bridge Server

Runs on the PC and accepts inference/training requests from the ARIA Android app
forwarded over an ADB USB tunnel.

Setup (run on the PC):
  1. Install dependencies:
       pip install llama-cpp-python flask

  2. Start the server:
       python aria_pc_server.py --model /path/to/model.gguf [--port 11435] [--gpu-layers 99]

  3. Set up the ADB TCP tunnel (in a separate terminal while phone is plugged in):
       adb forward tcp:11435 tcp:11435

  4. In the ARIA app: Settings → PC Bridge → tap "Connect"

The Android app sends requests to http://localhost:11435 which ADB transparently
forwards to this server on the PC. No Wi-Fi or network setup required.

Endpoints:
  GET  /health     — Liveness probe; returns server status and loaded model name
  POST /infer      — Run LLM inference on a prompt; returns JSON action string
  POST /sync       — Store experience tuples sent from the device
  GET  /adapter    — Download latest LoRA adapter weights to the device
  POST /train      — Run LoRA training on stored/sent experiences (optional)
"""

import argparse
import json
import os
import sys
import time
import threading
import logging
from pathlib import Path
from typing import Optional

from flask import Flask, request, jsonify, send_file, abort

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("aria_pc")

# ── App setup ──────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False

# ── Global state ───────────────────────────────────────────────────────────────
llm = None                    # llama_cpp.Llama instance
model_name: str = ""          # human-readable model filename
experiences: list = []        # in-memory experience buffer (also written to disk)
adapter_path: Optional[Path] = None  # path to the latest exported adapter
inference_lock = threading.Lock()    # prevent concurrent inference calls
data_dir = Path("aria_data")         # stores experiences.jsonl and adapters

# ── Model loading ──────────────────────────────────────────────────────────────

def load_model(model_path: str, n_gpu_layers: int, n_ctx: int) -> None:
    global llm, model_name
    try:
        from llama_cpp import Llama
        log.info(f"Loading model: {model_path}")
        log.info(f"  n_gpu_layers={n_gpu_layers}  n_ctx={n_ctx}")
        t0 = time.time()
        llm = Llama(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            n_threads=max(4, os.cpu_count() // 2),
            verbose=False,
        )
        model_name = Path(model_path).name
        elapsed = time.time() - t0
        log.info(f"Model loaded in {elapsed:.1f}s: {model_name}")
    except ImportError:
        log.error("llama-cpp-python not installed. Run: pip install llama-cpp-python")
        sys.exit(1)
    except Exception as exc:
        log.error(f"Failed to load model: {exc}")
        sys.exit(1)

# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    """Liveness probe — Android app calls this to check if the server is ready."""
    return jsonify({
        "status":    "ok",
        "model":     model_name,
        "n_exp":     len(experiences),
        "adapter":   str(adapter_path) if adapter_path else None,
    })


@app.route("/infer", methods=["POST"])
def infer():
    """
    Run LLM inference on the prompt from the Android app.

    Request body (JSON):
      {
        "prompt": "<full Llama chat-template prompt string>",
        "goal":   "<user goal — for logging only>",
        "max_tokens": 200   // optional, default 200
      }

    Response (JSON):
      {
        "action_json": "{\"tool\":\"Tap\",\"node_id\":\"#3\",\"reason\":\"...\"}"
      }
    """
    if llm is None:
        return jsonify({"error": "model not loaded"}), 503

    data = request.get_json(force=True, silent=True) or {}
    prompt     = data.get("prompt", "")
    goal       = data.get("goal", "")
    max_tokens = int(data.get("max_tokens", 200))

    if not prompt:
        return jsonify({"error": "prompt is required"}), 400

    log.info(f"Infer — goal: {goal[:60]!r}  prompt_len={len(prompt)}")

    with inference_lock:
        t0 = time.time()
        try:
            output = llm(
                prompt,
                max_tokens=max_tokens,
                stop=["<|eot_id|>", "<|end_of_text|>", "\n\n"],
                echo=False,
            )
            raw_text = output["choices"][0]["text"].strip()
            elapsed  = time.time() - t0
            tps      = output["usage"]["completion_tokens"] / max(elapsed, 0.001)
            log.info(f"Infer done in {elapsed:.2f}s  ({tps:.1f} tok/s)")
            log.debug(f"Raw output: {raw_text[:120]}")
        except Exception as exc:
            log.error(f"Inference error: {exc}")
            return jsonify({"error": str(exc)}), 500

    # Extract JSON action from raw text (same logic as PromptBuilder.parseAction)
    action_json = extract_json(raw_text)
    return jsonify({"action_json": action_json})


@app.route("/sync", methods=["POST"])
def sync():
    """
    Receive experience tuples from the Android app for training.

    Request body (JSON):
      { "experiences": [ {ExperienceTuple}, … ] }

    Response:
      { "stored": N }
    """
    data = request.get_json(force=True, silent=True) or {}
    new_exp = data.get("experiences", [])
    if not isinstance(new_exp, list):
        return jsonify({"error": "experiences must be a JSON array"}), 400

    experiences.extend(new_exp)
    data_dir.mkdir(exist_ok=True)
    with open(data_dir / "experiences.jsonl", "a") as f:
        for exp in new_exp:
            f.write(json.dumps(exp) + "\n")

    log.info(f"Stored {len(new_exp)} experience(s)  (total in-memory: {len(experiences)})")
    return jsonify({"stored": len(new_exp)})


@app.route("/adapter", methods=["GET"])
def adapter():
    """
    Download the latest LoRA adapter weights to the Android device.
    Returns the adapter binary if one has been produced by /train.
    """
    if adapter_path is None or not adapter_path.exists():
        abort(404, description="No adapter available — run /train first")
    return send_file(str(adapter_path), mimetype="application/octet-stream")


@app.route("/train", methods=["POST"])
def train():
    """
    Run LoRA fine-tuning on the accumulated experiences.

    Optional request body:
      { "experiences": [ {ExperienceTuple}, … ] }   // additional experiences to add

    Response:
      { "done": true, "loss": 0.12, "n_samples": 42 }

    NOTE: This requires a training framework (e.g. llama.cpp finetune or unsloth).
    The implementation here is a placeholder — wire in your preferred training
    library. The /adapter endpoint serves the resulting checkpoint.
    """
    global adapter_path

    data = request.get_json(force=True, silent=True) or {}
    extra = data.get("experiences", [])
    if extra:
        experiences.extend(extra)

    n = len(experiences)
    if n == 0:
        return jsonify({"error": "no experiences to train on"}), 400

    log.info(f"Training requested on {n} experience(s)")

    # ── Stub training ─────────────────────────────────────────────────────────
    # Replace this block with your actual fine-tuning code.
    # Options:
    #   • llama.cpp finetune CLI (fast LoRA on CPU/GPU)
    #   • unsloth (4× faster than HuggingFace, lower VRAM)
    #   • HuggingFace PEFT + TRL SFTTrainer
    #
    # The adapter should be written to: data_dir / "adapter.bin"
    # ──────────────────────────────────────────────────────────────────────────
    stub_adapter = data_dir / "adapter.bin"
    data_dir.mkdir(exist_ok=True)
    stub_adapter.write_bytes(b"ARIA_ADAPTER_STUB")
    adapter_path = stub_adapter
    log.warning("Training stub — write your training code in /train route")

    return jsonify({"done": True, "loss": 0.0, "n_samples": n})


# ── Helpers ────────────────────────────────────────────────────────────────────

def extract_json(text: str) -> str:
    """Extract the first {...} JSON object from LLM output text."""
    start = text.find("{")
    if start == -1:
        return '{"tool":"Wait","duration_ms":500,"reason":"no json in output"}'
    end = text.rfind("}")
    if end <= start:
        return '{"tool":"Wait","duration_ms":500,"reason":"malformed json"}'
    return text[start:end + 1].strip()


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="ARIA PC Bridge Server — offloads LLM inference from Android to PC over ADB"
    )
    parser.add_argument("--model",      required=True,        help="Path to the .gguf model file")
    parser.add_argument("--port",       type=int, default=11435, help="Server port (default: 11435)")
    parser.add_argument("--gpu-layers", type=int, default=99,    help="Number of GPU layers (default: 99 = all)")
    parser.add_argument("--ctx",        type=int, default=4096,  help="Context size (default: 4096)")
    args = parser.parse_args()

    load_model(args.model, args.gpu_layers, args.ctx)

    print(f"""
╔══════════════════════════════════════════════════════╗
║          ARIA PC Bridge Server — Ready               ║
╠══════════════════════════════════════════════════════╣
║  Model:     {args.model[:48]:<48}  ║
║  Port:      {args.port:<48}  ║
║  GPU layers:{args.gpu_layers:<48}  ║
╠══════════════════════════════════════════════════════╣
║  Next step: run in a new terminal:                   ║
║    adb forward tcp:{args.port} tcp:{args.port}                       ║
║  Then tap "Connect to PC" in ARIA Settings.          ║
╚══════════════════════════════════════════════════════╝
""")

    app.run(host="127.0.0.1", port=args.port, threaded=True)


if __name__ == "__main__":
    main()
