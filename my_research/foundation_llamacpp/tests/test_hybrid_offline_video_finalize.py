from my_research.foundation_llamacpp.runner import cli as runner_cli


PHASE_HEADER = (
    "row_type,elapsed_s_start,elapsed_s_end,rss_kb_start,rss_kb_end,"
    "col_a_ms,col_b_ms,total_ms,kv_pos,kv_total,kv_used_pct,"
    "kv_estimated_used_kb,kv_total_kb,kv_physical_committed_kb,token_idx\n"
)


def test_hybrid_offline_finalize_writes_phase_csv_when_phase_artifacts_exist(tmp_path, monkeypatch):
    result_dir = tmp_path / "InternVL3-2B-Instruct-Q8_0_hybrid_ctx_4096_kv16"
    result_dir.mkdir()
    (result_dir / "hybrid_decode_stdout.txt").write_text(
        "llama_perf_context_print:        load time =    10.00 ms\n",
        encoding="utf-8",
    )
    (result_dir / "foundation_exit_code.txt").write_text("0", encoding="utf-8")
    (result_dir / "decoder_phase_stats.csv").write_text(
        PHASE_HEADER + "L_DecoderLoad,0.000000,0.010000,,,10,,10,,,,,,,0\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(runner_cli, "_write_memory_usage_txt", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runner_cli, "_write_png_memory_timeline", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runner_cli, "_write_png_phase_duration_from_rows", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runner_cli, "_write_png_phase_duration", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runner_cli, "_write_png_phase_timeline", lambda *_args, **_kwargs: None)

    runner_cli._finalize_hybrid_outputs(result_dir)

    proc = result_dir / "foundation_proc.csv"
    assert proc.exists()
    assert "L_DecoderLoad" in proc.read_text(encoding="utf-8")
