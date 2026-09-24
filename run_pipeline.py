"""Ejecuta notebooks, entrenamiento, monitoreo y pruebas, deteniéndose si algo falla."""
import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def execute_notebooks(in_process=False):
    import nbformat
    if in_process:
        # Alternativa sin sockets para entornos donde no puede iniciarse un kernel.
        from IPython.core.interactiveshell import InteractiveShell
        from IPython.utils.capture import capture_output
        import matplotlib
        matplotlib.use("module://matplotlib_inline.backend_inline")
        from matplotlib_inline.backend_inline import show, set_matplotlib_formats
        shell = InteractiveShell.instance()
        set_matplotlib_formats("png")
        for name in ["Cargar_datos.ipynb", "comprension_eda.ipynb"]:
            path = ROOT / "src" / name
            notebook = nbformat.read(path, as_version=4)
            namespace = {"__name__": "__main__"}
            count = 0
            for cell in notebook.cells:
                if cell.cell_type != "code":
                    continue
                count += 1
                with capture_output() as captured:
                    exec(compile(cell.source, name, "exec"), namespace)
                    show(close=True)
                cell.execution_count = count
                cell.outputs = []
                for stream, text in [("stdout", captured.stdout), ("stderr", captured.stderr)]:
                    if text:
                        cell.outputs.append(nbformat.v4.new_output("stream", name=stream, text=text))
                for item in captured.outputs:
                    cell.outputs.append(nbformat.v4.new_output("display_data", data=item.data, metadata=item.metadata))
            nbformat.validate(notebook)
            nbformat.write(notebook, path)
            print(f"Notebook ejecutado en proceso: {name}", flush=True)
        return
    from nbclient import NotebookClient
    import json
    # Kernel temporal: garantiza usar el mismo Python que ejecuta este script.
    with tempfile.TemporaryDirectory() as folder:
        directory = Path(folder) / "kernels/credit-local"
        directory.mkdir(parents=True)
        (directory / "kernel.json").write_text(json.dumps({
            "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            "display_name": "Credit local", "language": "python"}), encoding="utf-8")
        previous = os.environ.get("JUPYTER_PATH")
        os.environ["JUPYTER_PATH"] = folder + (os.pathsep + previous if previous else "")
        try:
            for name in ["Cargar_datos.ipynb", "comprension_eda.ipynb"]:
                path = ROOT / "src" / name
                notebook = nbformat.read(path, as_version=4)
                NotebookClient(notebook, timeout=180, kernel_name="credit-local",
                               resources={"metadata": {"path": str(ROOT)}}).execute()
                nbformat.validate(notebook)
                nbformat.write(notebook, path)
                print(f"Notebook ejecutado: {name}", flush=True)
        finally:
            if previous is None:
                os.environ.pop("JUPYTER_PATH", None)
            else:
                os.environ["JUPYTER_PATH"] = previous


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-notebooks", action="store_true")
    parser.add_argument("--in-process", action="store_true", help="Ejecutar celdas Python sin kernel de red")
    args = parser.parse_args()
    if not args.skip_notebooks:
        execute_notebooks(args.in_process)
    for command in [["-m", "src.model_training_evaluation"], ["-m", "src.model_monitoring"],
                    ["-m", "pytest", "src/test_project.py", "-q", "--cov=src", "--cov-report=term-missing", "--cov-report=xml"]]:
        subprocess.run([sys.executable, *command], cwd=ROOT, check=True)
    print("Pipeline completado. Resultados guardados en src/.")


if __name__ == "__main__":
    main()
