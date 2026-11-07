#!/usr/bin/env python3
"""
FoxDot AI TUI
- Tab 1: Chat with the current FoxDot model + Ollama service controls
- Tab 2: Fine-tune (drop new examples, auto-train, auto-export, auto-load)
- Tab 3: System console (all errors, warnings, events)

Run: python foxdot_tui.py
"""

import json
import logging
import os
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import ollama
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import (
    Button,
    Footer,
    Header,
    Input,
    Label,
    ProgressBar,
    RichLog,
    Static,
    TabbedContent,
    TabPane,
)

# ─── Config ───────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
NEW_EXAMPLES_DIR = DATA_DIR / "new_examples"
FULL_DATASET = DATA_DIR / "full_dataset.jsonl"
ADAPTERS_DIR = BASE_DIR / "adapters"
GGUF_DIR = BASE_DIR / "gguf"
MODELFILE = BASE_DIR / "Modelfile"

BASE_MODEL = "unsloth/Qwen3-4B-bnb-4bit"
OLLAMA_MODEL_NAME = "foxdot"
SYSTEM_PROMPT = (
    "You are a FoxDot live-coding assistant. "
    "Always generate complete, runnable FoxDot code using with synths: / with noise: / with bass: blocks. "
    "Use PSequence, PChoice, PWeightedChoice, PRand, PArray patterns. "
    "Include bpm, dur, swing, and filter parameters. "
    "Use only valid FoxDot API calls (pluck, sawtooth, triangle, sine, noise, bass, kick, snare, hat)."
)

# Training hyperparameters
MAX_SEQ_LENGTH = 2048
BATCH_SIZE = 1
GRAD_ACCUM = 8
EPOCHS_INITIAL = 3
EPOCHS_INCREMENTAL = 2
LR_INITIAL = 2e-4
LR_INCREMENTAL = 1e-4
LORA_R = 16
LORA_ALPHA = 16
REHEARSAL_RATIO = 0.25


# ─── Console Logger ──────────────────────────────────────────────────────────

class ConsoleHandler(logging.Handler):
    """Routes log records to the TUI console tab."""

    def __init__(self, app):
        super().__init__()
        self.app = app

    def emit(self, record):
        try:
            msg = self.format(record)
            self.app.call_from_thread(self._write, msg, record.levelno)
        except Exception:
            pass

    def _write(self, msg: str, levelno: int) -> None:
        try:
            console = self.app.query_one("#console-log", RichLog)
            if levelno >= logging.ERROR:
                console.write(f"[red]{msg}[/red]")
            elif levelno >= logging.WARNING:
                console.write(f"[yellow]{msg}[/yellow]")
            else:
                console.write(msg)
        except Exception:
            pass


def setup_logging(app) -> None:
    """Configure logging to write to the TUI console."""
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    handler = ConsoleHandler(app)
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)-7s] %(name)s: %(message)s",
                          datefmt="%H:%M:%S")
    )
    logger.addHandler(handler)

    # Suppress noisy libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("datasets").setLevel(logging.WARNING)


def log_event(msg: str, level: int = logging.INFO) -> None:
    """Convenience: log to the console tab."""
    logging.getLogger("foxdot").log(level, msg)


# ─── Ollama service helpers ───────────────────────────────────────────────────

def ollama_is_running() -> bool:
    try:
        client = ollama.Client()
        client.list()
        return True
    except Exception:
        return False


def ollama_service_start() -> str:
    result = subprocess.run(
        ["systemctl", "start", "ollama"], capture_output=True, text=True
    )
    if result.returncode == 0:
        return "Started via systemctl."
    try:
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        time.sleep(2)
        if ollama_is_running():
            return "Started (background process)."
        return "Started — still initializing."
    except Exception as e:
        return f"Failed to start: {e}"


def ollama_service_stop() -> str:
    result = subprocess.run(
        ["systemctl", "stop", "ollama"], capture_output=True, text=True
    )
    if result.returncode == 0:
        return "Stopped via systemctl."
    subprocess.run(["pkill", "-f", "ollama serve"], capture_output=True)
    return "Stopped (pkill)."


def ollama_service_restart() -> str:
    subprocess.run(["systemctl", "stop", "ollama"], capture_output=True)
    subprocess.run(["pkill", "-f", "ollama serve"], capture_output=True)
    time.sleep(1)
    result = subprocess.run(
        ["systemctl", "start", "ollama"], capture_output=True, text=True
    )
    if result.returncode == 0:
        return "Restarted via systemctl."
    try:
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        time.sleep(2)
        return "Restarted (background process)."
    except Exception as e:
        return f"Failed to restart: {e}"


# ─── Helper functions ─────────────────────────────────────────────────────────

def get_latest_version() -> int:
    if not ADAPTERS_DIR.exists():
        return 0
    versions = [
        int(d.name[1:])
        for d in ADAPTERS_DIR.iterdir()
        if d.is_dir() and d.name.startswith("v") and d.name[1:].isdigit()
    ]
    return max(versions) if versions else 0


def load_jsonl(path: Path) -> list:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_jsonl(rows: list, path: Path) -> None:
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def collect_new_examples() -> list:
    new_rows = []
    if not NEW_EXAMPLES_DIR.exists():
        return new_rows
    for f in sorted(NEW_EXAMPLES_DIR.glob("*.jsonl")):
        new_rows.extend(load_jsonl(f))
    return new_rows


def build_training_set(new_rows: list, version: int) -> list:
    if version == 0:
        return new_rows
    old_rows = load_jsonl(FULL_DATASET) if FULL_DATASET.exists() else []
    import random
    n_rehearsal = max(1, int(len(old_rows) * REHEARSAL_RATIO))
    rehearsal = random.sample(old_rows, min(n_rehearsal, len(old_rows)))
    return rehearsal + new_rows


def run_training(training_data, version, status_callback, log_callback):
    from unsloth import FastLanguageModel
    from datasets import Dataset
    from trl import SFTTrainer
    from transformers import TrainingArguments

    is_incremental = version > 0
    prev_adapter = ADAPTERS_DIR / f"v{version}" if is_incremental else None

    log_callback(f"Loading base model: {BASE_MODEL}")
    if is_incremental:
        log_callback(f"Loading previous adapter: {prev_adapter}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(prev_adapter),
            max_seq_length=MAX_SEQ_LENGTH,
            load_in_4bit=True,
        )
    else:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=BASE_MODEL,
            max_seq_length=MAX_SEQ_LENGTH,
            load_in_4bit=True,
        )

    log_callback("Applying LoRA...")
    model = FastLanguageModel.get_peft_model(
        model, r=LORA_R, lora_alpha=LORA_ALPHA,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        use_gradient_checkpointing="unsloth",
    )

    dataset = Dataset.from_list(training_data)
    log_callback(f"Training on {len(dataset)} examples...")

    output_dir = ADAPTERS_DIR / f"v{version + 1}"
    output_dir.mkdir(parents=True, exist_ok=True)

    log_callback("Starting training...")
    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer,
        train_dataset=dataset,
        args=TrainingArguments(
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACCUM,
            num_train_epochs=EPOCHS_INCREMENTAL if is_incremental else EPOCHS_INITIAL,
            learning_rate=LR_INCREMENTAL if is_incremental else LR_INITIAL,
            output_dir=str(output_dir),
            logging_steps=5,
            save_strategy="no",
            report_to="none",
        ),
    )
    trainer.train()

    log_callback("Saving adapter...")
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    gguf_out = GGUF_DIR / f"v{version + 1}"
    gguf_out.mkdir(parents=True, exist_ok=True)
    log_callback("Exporting to GGUF (q4_k_m)...")
    model.save_pretrained_gguf(str(gguf_out), tokenizer, quantization_method=["q4_k_m"])

    gguf_files = list(gguf_out.glob("*.gguf"))
    if not gguf_files:
        raise RuntimeError("GGUF export produced no files")
    gguf_path = gguf_files[0]

    log_callback("Creating Ollama model...")
    modelfile_content = (
        f"FROM {gguf_path}\n"
        f'SYSTEM "{SYSTEM_PROMPT}"\n'
        f"PARAMETER temperature 0.3\n"
        f"PARAMETER num_ctx 4096\n"
    )
    MODELFILE.write_text(modelfile_content)

    result = subprocess.run(
        ["ollama", "create", OLLAMA_MODEL_NAME, "-f", str(MODELFILE)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ollama create failed: {result.stderr}")

    log_callback(f"✅ Done! Model '{OLLAMA_MODEL_NAME}' is ready (v{version + 1}).")
    status_callback("done")


# ─── The TUI App ──────────────────────────────────────────────────────────────

class FoxDotTUI(App):
    CSS = """
    Header, Footer {
        background: $primary;
    }
    #chat-log {
        height: 1fr;
        border: round $accent;
    }
    #chat-input {
        dock: bottom;
    }
    #ollama-controls {
        dock: bottom;
        padding: 0 1;
        height: 3;
    }
    #train-log {
        height: 1fr;
        border: round $success;
    }
    #train-controls {
        dock: bottom;
        padding: 1 0;
    }
    #train-info {
        padding: 0 1;
    }
    #console-log {
        height: 1fr;
        border: round $error;
    }
    #console-toolbar {
        dock: bottom;
        padding: 0 1;
    }
    Button {
        margin: 0 1;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("tab", "focus_next", "Next tab"),
        ("shift+tab", "focus_previous", "Prev tab"),
    ]

    def __init__(self):
        super().__init__()
        self.chat_history: list = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
        self.is_training = False

    def compose(self) -> ComposeResult:
        yield Header()
        with TabbedContent(initial="chat"):
            with TabPane("💬 Chat", id="chat"):
                yield RichLog(id="chat-log", wrap=True, markup=True)
                with Horizontal(id="chat-input"):
                    yield Input(
                        placeholder="Ask about FoxDot code...",
                        id="chat-msg",
                    )
                    yield Button("Send", id="chat-send", variant="primary")
                with Horizontal(id="ollama-controls"):
                    yield Static("Ollama:", id="ollama-label")
                    yield Static("●", id="ollama-status")
                    yield Button("▶ Start", id="ollama-start")
                    yield Button("↻ Restart", id="ollama-restart")
                    yield Button("■ Stop", id="ollama-stop")
                    yield Button("🔄 Check", id="ollama-check")

            with TabPane("🔧 Fine-tune", id="train"):
                yield Static(
                    f"Latest adapter: v{get_latest_version()}  |  "
                    f"New examples in: {NEW_EXAMPLES_DIR}",
                    id="train-info",
                )
                yield RichLog(id="train-log", wrap=True)
                with Horizontal(id="train-controls"):
                    yield Button("📂 Scan & Train", id="train-btn", variant="success")
                    yield Button("🧹 Clear new examples", id="clear-btn")
                yield ProgressBar(total=100, show_eta=False, id="train-progress")

            with TabPane("🖥 Console", id="console"):
                yield RichLog(id="console-log", wrap=True, markup=True)
                with Horizontal(id="console-toolbar"):
                    yield Button("🧹 Clear", id="console-clear")
                    yield Button("💾 Save to file", id="console-save")
        yield Footer()

    def on_mount(self) -> None:
        # Set up logging AFTER widgets exist
        setup_logging(self)
        log_event("FoxDot TUI started.", logging.INFO)
        log_event(f"Python {sys.version}", logging.DEBUG)
        log_event(f"Working dir: {BASE_DIR}", logging.DEBUG)

        self.query_one("#chat-log", RichLog).write(
            "[bold]FoxDot AI[/bold] — type a message below. "
            f"Model: [green]{OLLAMA_MODEL_NAME}[/green]\n"
        )
        self.query_one("#chat-msg", Input).focus()
        self._update_ollama_status()

    def _update_ollama_status(self) -> None:
        status = self.query_one("#ollama-status", Static)
        if ollama_is_running():
            status.update("[green]●[/green] Running")
            log_event("Ollama: running", logging.INFO)
        else:
            status.update("[red]●[/red] Stopped")
            log_event("Ollama: NOT running", logging.WARNING)

    # ─── Button / Input handlers ─────────────────────────────────────────────

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "chat-send":
            await self.send_chat_message()
        elif event.button.id == "train-btn":
            await self.start_training()
        elif event.button.id == "clear-btn":
            self.clear_new_examples()
        elif event.button.id == "ollama-start":
            self._ollama_action("start")
        elif event.button.id == "ollama-restart":
            self._ollama_action("restart")
        elif event.button.id == "ollama-stop":
            self._ollama_action("stop")
        elif event.button.id == "ollama-check":
            self._update_ollama_status()
        elif event.button.id == "console-clear":
            self.query_one("#console-log", RichLog).clear()
        elif event.button.id == "console-save":
            self._save_console()

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "chat-msg":
            await self.send_chat_message()

    # ─── Ollama service controls ─────────────────────────────────────────────

    def _ollama_action(self, action: str) -> None:
        chat_log = self.query_one("#chat-log", RichLog)

        if action == "start":
            msg = ollama_service_start()
        elif action == "stop":
            msg = ollama_service_stop()
        elif action == "restart":
            msg = ollama_service_restart()
        else:
            return

        chat_log.write(f"[bold yellow]Ollama:[/bold yellow] {msg}")
        log_event(f"Ollama {action}: {msg}", logging.INFO)
        time.sleep(1)
        self._update_ollama_status()

    # ─── Chat ─────────────────────────────────────────────────────────────────

    async def send_chat_message(self) -> None:
        msg_input = self.query_one("#chat-msg", Input)
        text = msg_input.value.strip()
        if not text:
            return

        with msg_input.prevent(Input.Changed):
            msg_input.value = ""

        chat_log = self.query_one("#chat-log", RichLog)
        chat_log.write(f"[bold blue]You:[/bold blue] {text}")
        log_event(f"Chat → {text[:80]}", logging.DEBUG)

        self.chat_history.append({"role": "user", "content": text})

        try:
            response = ollama.chat(
                model=OLLAMA_MODEL_NAME,
                messages=self.chat_history,
                options={"temperature": 0.3, "num_ctx": 4096},
            )
            reply = response["message"]["content"]
            self.chat_history.append({"role": "assistant", "content": reply})
            chat_log.write(f"[bold green]FoxDot AI:[/bold green]\n{reply}\n")
            log_event(f"Chat ← {len(reply)} chars", logging.DEBUG)
        except Exception as e:
            chat_log.write(f"[bold red]Error:[/bold red] {e}")
            log_event(f"Chat error: {e}", logging.ERROR)
            self.chat_history.pop()

        msg_input.focus()

    # ─── Fine-tune ────────────────────────────────────────────────────────────

    def clear_new_examples(self) -> None:
        if NEW_EXAMPLES_DIR.exists():
            for f in NEW_EXAMPLES_DIR.glob("*.jsonl"):
                f.unlink()
        self.query_one("#train-log", RichLog).write("[yellow]Cleared new_examples/.[/yellow]")
        log_event("Cleared new_examples/", logging.INFO)

    async def start_training(self) -> None:
        if self.is_training:
            return

        new_rows = collect_new_examples()
        if not new_rows:
            self.query_one("#train-log", RichLog).write(
                "[red]No new examples found in data/new_examples/.[/red]"
            )
            log_event("Training aborted: no new examples found", logging.WARNING)
            return

        version = get_latest_version()
        is_incremental = version > 0
        training_data = build_training_set(new_rows, version)

        existing = load_jsonl(FULL_DATASET) if FULL_DATASET.exists() else []
        existing.extend(new_rows)
        save_jsonl(existing, FULL_DATASET)

        self.is_training = True
        self.query_one("#train-btn", Button).disabled = True
        self.query_one("#train-info", Static).update(
            f"Training v{version + 1}... ({len(training_data)} examples, "
            f"{'incremental' if is_incremental else 'initial'})"
        )

        log = self.query_one("#train-log", RichLog)
        log.write(
            f"[bold]Starting v{version + 1} training...[/bold]\n"
            f"  New examples: {len(new_rows)}\n"
            f"  Rehearsal: {len(training_data) - len(new_rows)}\n"
            f"  Total: {len(training_data)}\n"
            f"  LR: {LR_INCREMENTAL if is_incremental else LR_INITIAL}\n"
            f"  Epochs: {EPOCHS_INCREMENTAL if is_incremental else EPOCHS_INITIAL}\n"
        )
        log_event(f"Training v{version+1} started: {len(training_data)} examples", logging.INFO)

        def _status_cb(state: str):
            self.call_from_thread(self._on_train_status, state)

        def _log_cb(msg: str):
            self.call_from_thread(
                lambda m=msg: self.query_one("#train-log", RichLog).write(m)
            )
            self.call_from_thread(log_event, m, logging.DEBUG)

        try:
            self.run_worker(
                lambda: run_training(training_data, version, _status_cb, _log_cb),
                name="training",
                exclusive=True,
                thread=True,
            )
        except Exception as e:
            log_event(f"Training worker failed to start: {e}", logging.ERROR)
            self.is_training = False
            self.query_one("#train-btn", Button).disabled = False

    def _on_train_status(self, state: str) -> None:
        if state == "done":
            self.is_training = False
            self.query_one("#train-btn", Button).disabled = False
            v = get_latest_version()
            self.query_one("#train-info", Static).update(
                f"✅ Latest adapter: v{v}  |  Model ready in Ollama."
            )
            for f in NEW_EXAMPLES_DIR.glob("*.jsonl"):
                f.unlink()
            self.query_one("#train-log", RichLog).write(
                "\n[yellow]New examples cleared. Drop more to train again.[/yellow]"
            )
            log_event(f"Training v{v} completed successfully.", logging.INFO)

    # ─── Console ──────────────────────────────────────────────────────────────

    def _save_console(self) -> None:
        """Save console contents to a file."""
        console = self.query_one("#console-log", RichLog)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = BASE_DIR / f"console_{ts}.txt"
        try:
            # RichLog stores lines in .lines (list of Rich segments)
            text = "\n".join(str(line) for line in console.lines)
            path.write_text(text)
            log_event(f"Console saved to {path}", logging.INFO)
        except Exception as e:
            log_event(f"Failed to save console: {e}", logging.ERROR)

    # ─── Global error handler ─────────────────────────────────────────────────

    def on_exception(self, event) -> None:
        """Catch all unhandled exceptions and log them to the console."""
        log_event(
            f"UNHANDLED EXCEPTION: {event.exception}\n"
            f"{traceback.format_exception(type(event.exception), event.exception, event.exception.__traceback__)}",
            logging.ERROR,
        )
        super().on_exception(event)


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    FoxDotTUI().run()   
