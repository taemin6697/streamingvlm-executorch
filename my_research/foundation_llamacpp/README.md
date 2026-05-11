# for_cursor_llm_llamacpp

This folder holds the llama.cpp-side runner and helper notes for the foundation work.
The active structured refactor log lives in `docs/for_cursor_llm_llamacpp_version2.md`;
the older cumulative log is archived at `docs/archive/for_cursor_llm_llamacpp.md`.
The actual run outputs should be written under:

```text
my_research/foundation_llamacpp/results/log/<backend>/<model_name>/
```

Use the GGUF stem as `model_name`, for example `InternVL3-1B-Instruct-Q8_0`,
so the quantization suffix stays visible in the result path.

Hybrid bridge prototype:

- Source lives under `hybrid_bridge/` and stays outside upstream `llama.cpp` and
  ExecuTorch.
- `hybrid_vision_dump` runs the ExecuTorch QNN vision encoder and writes a
  float32 embedding file.
- `hybrid_decode` runs llama.cpp with OpenCL and injects that external
  embedding through the public mtmd helper path.
- `run_android_hybrid_bridge.py` orchestrates the split process on Android.

Use the runner to collect:

- control output from `llama.cpp` or hybrid runtime runs
- memory breakdowns
- total runtime
- per-stage timing such as image encode, prefill, and decode
- backend-specific observations for CPU, Vulkan, OpenCL, or other paths

Expected output files in each result folder:

- `foundation_output.txt`
- `foundation_exit_code.txt`
- `foundation_proc.csv`
- `android_memory_timeline.csv`
- `vision_output_stats.csv` when timing data is available

Keep raw binaries, build outputs, and other large artifacts out of this folder.
