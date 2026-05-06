# streamingvlm-executorch

StreamingVLM on Mobile — research workspace (ExecuTorch, **vendored `llama.cpp/`**, and hybrid tooling under `my_research/`).

- **`llama.cpp/`** is tracked in this repo (**no nested `.git`**). Sync with upstream with patches or manual `git remote` workflows as needed.
- **`llama.cpp/models/`** is **not committed** (see root `.gitignore`). Keep GGUF/MMProj on device or CI cache.
- **Prior nested `llama.cpp` Git history** (if you made one) may be archived as `llama.cpp-nested-history.bundle` locally; it is gitignored.
