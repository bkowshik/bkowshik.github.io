---
title: "How I use Google Colab"
description: "Notes and recipes from my day-to-day Google Colab workflow — setup, data handling, and package installs."
pubDatetime: 2026-06-10T09:00:00+05:30
tags:
  - tools
---

Notes and recipes from how I use [Google Colab](https://colab.research.google.com/) day to day — setup, data handling, and package installs. All snippets are meant to be run inside Colab cells — lines starting with `!` are shell commands run from the notebook, and `%config` is an IPython magic.

## 💡 Why Google Colab

The choice of Colab as the default substrate is deliberate — the principles below explain why.

### 🌐 Data lives near the compute

Datasets for serious work are large — archives routinely run into tens or hundreds of GB. Downloading them to a laptop is bandwidth-bound and wastes local disk. On Colab the data is pulled directly into a cloud VM that sits next to Google's network, so the same download is faster and free of local storage pressure.

### ⚡ Free GPU / TPU runtime

Foundation models and most modern deep learning pipelines need a GPU. Colab provides one for free, with TPU and higher-tier GPUs available on paid plans. There is no driver install, no CUDA mismatch — the runtime comes pre-configured.

### 🧰 Zero local setup

Python, PyTorch, JAX, NumPy, SciPy, scikit-learn, and most of the scientific stack ship pre-installed. The notebook environment is identical across machines, which removes the "works on my laptop" problem and makes recipes reproducible.

### 🔗 Shareable by default

A Colab notebook is a single Drive file. It can be opened, copied, or shared with a link, which makes collaboration with colleagues, students, and reviewers trivial — no local environment to replicate.

### 🫧 Ephemeral, by design

Runtimes are wiped on disconnect. This forces every notebook to be re-runnable from a clean state, which is exactly the discipline you want for scientific work. Persistent state goes to Google Drive; everything else is reproducible from the cell.

## 📂 Mount Google Drive

Drive is the only storage that survives a runtime disconnect, so mounting it is usually the first cell:

```python
from google.colab import drive
drive.mount('/content/drive')
```

## 🗄️ Filesystem

- `/content` is the default working directory and where you land when a notebook starts. It typically has around `100+ GB` of free disk space (varies by runtime type). Anything here is ephemeral and gets wiped when the runtime disconnects.
- Home directory `/root` is writable too, but `/content` is the convention.
- `/tmp` is also writable and ephemeral, but smaller. Good for temporary scratch files.
- `/content/drive/MyDrive` is available after you mount Google Drive. Files here persist across sessions since they live in your Drive (subject to your Drive quota).

```bash
!df -h /content
```

```
Filesystem      Size  Used Avail Use% Mounted on
overlay         108G   22G   86G  21% /
```

## 💾 Copy data to personal Google Drive

Downloaded datasets live in ephemeral `/content` and vanish with the runtime. Sync them to Drive once so the next session can skip the download:

```bash
!rsync -ah --info=progress2 --stats /content/data/ /content/drive/MyDrive/data/
```

## 🔑 Set environment variables

Some datasets and APIs are password-protected; libraries usually pick the credentials up from environment variables, so set them at the top of the notebook before anything else runs:

```python
import os
os.environ["DATASET_PASSWORD"] = "xxx"
```

## 📦 Install packages

From PyPI (recommended — the latest tagged release):

```bash
%pip install --upgrade "package[all]"
```

From the GitHub main branch (for unreleased changes):

```bash
%pip install --upgrade "package[all] @ git+https://github.com/org/repo.git#subdirectory=package-repo"
```

### Why `--upgrade`

Colab images come with a large set of pre-installed packages — NumPy `2.0.2`, for example. Without `--upgrade`, pip's resolver leaves any already-installed package in place as long as it nominally satisfies a requirement, even when the package you are installing and its transitive deps (SciPy, scikit-learn) were built against a newer NumPy. The mismatch surfaces deep in the import chain:

```
ImportError: cannot import name '_center' from 'numpy._core.umath'
```

With `--upgrade`, pip honors the version constraints declared in the package's `pyproject.toml` and bumps NumPy (and anything else) to a compatible version. ⚠️ After the install finishes, restart the Colab runtime once so the kernel picks up the new binaries.

## ✨ Retina plots

One line at the top of the notebook makes matplotlib figures render crisp on high-DPI displays:

```python
%config InlineBackend.figure_format = 'retina'
```

## 🧩 VS Code Colab extension

The [Google Colab VS Code extension](https://github.com/googlecolab/colab-vscode) connects VS Code to Colab's hosted Jupyter runtime, giving access to free GPUs and TPUs without leaving the editor.

## 🐛 Plotly charts not rendering

`fig.show()` can silently render nothing in the VS Code Colab extension. Plotly's auto-detection picks its `colab` renderer whenever the `google.colab` module is importable — even outside web Colab — and that renderer's output only displays in the Colab web app ([plotly.py#5471](https://github.com/plotly/plotly.py/issues/5471)).

Two fixes:

1. Upgrade plotly — from `6.8.0` detection uses the `COLAB_NOTEBOOK_ID` environment variable (set only in web Colab) instead of the module import ([plotly.py#5473](https://github.com/plotly/plotly.py/pull/5473)).

   ```bash
   %pip install -U plotly
   ```

2. Or set the renderer explicitly at the top of the notebook (works on any version):

   ```python
   import plotly.io as pio
   pio.renderers.default = "vscode"
   ```

---

This is a living list — I'll keep adding recipes as my workflow evolves. 🌱
