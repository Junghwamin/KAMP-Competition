"""Run the full notebook with a local kernel and durable progress checkpoints."""
from __future__ import annotations

import json
import asyncio
import os
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]


def main():
    os.chdir(ROOT)
    os.environ.update(PYTHONHASHSEED="42", PYTHONIOENCODING="utf-8", KAMP_FAST="0")
    cache = ROOT / ".cache" / "full-model-run"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["JUPYTER_PATH"] = str(cache / "jupyter")
    os.environ["JUPYTER_RUNTIME_DIR"] = str(cache / "runtime")
    os.environ["IPYTHONDIR"] = str(cache / "ipython")
    os.environ["MPLCONFIGDIR"] = str(cache / "matplotlib")
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    kernel = cache / "jupyter" / "kernels" / "kamp-full"
    kernel.mkdir(parents=True, exist_ok=True)
    (kernel / "kernel.json").write_text(json.dumps({
        "argv": [sys.executable, "-X", "utf8", "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": "KAMP full training", "language": "python",
    }), encoding="utf-8")
    import nbformat
    from nbclient import NotebookClient

    def execute(path, checkpoint):
        nb = nbformat.read(path, as_version=4)
        def started(cell, cell_index, **kwargs):
            if cell.cell_type == "code":
                print(f"{datetime.now().isoformat(timespec='seconds')} CELL {cell_index + 1}/{len(nb.cells)} {cell.source.splitlines()[0][:100]}", flush=True)
        def completed(cell, cell_index, **kwargs):
            nbformat.write(nb, checkpoint)
            for output in cell.get("outputs", []):
                if output.output_type == "stream" and output.name == "stdout":
                    print(output.text[-2500:], flush=True)
        client = NotebookClient(nb, timeout=7200, kernel_name="kamp-full",
            resources={"metadata": {"path": str(ROOT)}},
            on_cell_start=started, on_cell_executed=completed)
        try:
            client.execute()
        finally:
            nbformat.write(nb, checkpoint)
        nbformat.write(nb, path)

    if "--check-only" in sys.argv:
        execute(ROOT / "START_HERE.ipynb", cache / "environment.ipynb")
        return
    backup = cache / "before"
    if not backup.exists():
        backup.mkdir()
        shutil.copytree(ROOT / "outputs", backup / "outputs", ignore=shutil.ignore_patterns("models"))
        shutil.copy2(ROOT / "자원최적화_제안모델.ipynb", backup)
    subprocess.run([sys.executable, "-X", "utf8", "tools/build_notebook.py"], check=True)
    execute(ROOT / "자원최적화_제안모델.ipynb", cache / "checkpoint.ipynb")
    subprocess.run([sys.executable, "-X", "utf8", "tools/finalize_notebook.py"], check=True)
    for role in ("eval", "deploy"):
        subprocess.run([sys.executable, "-X", "utf8", "-m", "serving", "verify",
                        "--bundle", f"outputs/models/full/{role}"], check=True)
    print("FULL_TRAINING_AND_BUNDLE_VERIFICATION_COMPLETE", flush=True)


if __name__ == "__main__":
    if "--log" in sys.argv:
        logdir = ROOT / ".cache" / "full-model-run"
        logdir.mkdir(parents=True, exist_ok=True)
        with (logdir / "run.log").open("w", encoding="utf-8", buffering=1) as log:
            os.dup2(log.fileno(), 1)
            os.dup2(log.fileno(), 2)
            main()
    else:
        main()
