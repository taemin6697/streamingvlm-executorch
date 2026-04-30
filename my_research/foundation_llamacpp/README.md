# for_cursor_llm_llamacpp

This folder holds the llama.cpp-side runner and helper notes for the foundation work.
The detailed cumulative log lives in `docs/for_cursor_llm_llamacpp.md`.
The actual run outputs should be written under:

```text
my_research/foundation_llamacpp/results/log/<backend>/<model_name>/
```

Use the GGUF stem as `model_name`, for example `InternVL3-1B-Instruct-Q8_0`,
so the quantization suffix stays visible in the result path.

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
