import pytest
from pathlib import Path

from my_research.foundation_llamacpp.runner import cli as runner_cli


ROOT = Path(__file__).resolve().parents[3]
HYBRID_VISION_DUMP = ROOT / "my_research/foundation_llamacpp/hybrid_bridge/hybrid_vision_dump.cpp"
HYBRID_DECODE = ROOT / "my_research/foundation_llamacpp/hybrid_bridge/hybrid_decode.cpp"


def _row(name: str, start: float, end: float) -> dict[str, str]:
    return {
        "row_type": name,
        "elapsed_s_start": f"{start:.6f}",
        "elapsed_s_end": f"{end:.6f}",
        "total_ms": f"{(end - start) * 1000.0:.0f}",
        "token_idx": "",
    }


def test_phase_duration_breakdown_uses_common_labels_and_only_aggregate_dynamic_kv():
    rows = [
        _row("ImagePrefill", 0.0, 1.0),
        _row("D", 1.0, 1.2),
        _row("DynamicKVGrow", 1.2, 1.4),
        _row("DynamicKVGrowCopy", 1.25, 1.30),
    ]

    phases = runner_cli._phase_duration_breakdown(rows)

    assert phases == [
        ("ImagePrefill", pytest.approx(1.0)),
        ("DynamicKVGrow", pytest.approx(0.2)),
        ("Decode", pytest.approx(0.2)),
    ]


def test_common_phase_timeline_writer_outputs_phase_timeline_for_offline_rows(tmp_path):
    rows = [
        _row("V_Encode", 0.0, 0.1),
        _row("ImagePrefill", 0.1, 0.3),
        _row("T_Prefill", 0.3, 0.4),
        _row("D", 0.4, 0.5),
    ]

    runner_cli._write_png_phase_timeline(tmp_path, rows)

    assert (tmp_path / "phase_timeline.png").exists()
    assert not (tmp_path / "streaming_phase_timeline.png").exists()


def test_common_phase_timeline_writer_outputs_phase_timeline_for_stream_rows(tmp_path):
    (tmp_path / "stream_events.csv").write_text(
        "event,elapsed_s_start,video_time_s\nStreamFrameEnqueue,10.000000,0.000000\n",
        encoding="utf-8",
    )
    rows = [
        _row("StreamPromptPrefill", 15.0, 15.0),
        _row("VisionPrefillV_Encode", 15.0, 15.1),
        _row("VisionPrefillMmproj", 15.1, 15.2),
        _row("VisionPrefillImagePrefill", 15.2, 15.3),
        _row("VisionPrefillT_Prefill", 15.3, 15.4),
        _row("D", 15.4, 15.5),
    ]

    runner_cli._write_png_phase_timeline(tmp_path, rows, stream_time=True)

    assert (tmp_path / "phase_timeline.png").exists()
    assert not (tmp_path / "streaming_phase_timeline.png").exists()


def test_offline_phase_timeline_rebases_after_hidden_setup(tmp_path):
    rows = [
        _row("L_DecoderLoad", 0.0, 20.0),
        _row("Mmproj", 20.0, 21.0),
        _row("ImagePrefill", 21.0, 22.0),
        _row("D", 22.0, 23.0),
    ]

    phases, _markers, origin, end = runner_cli._phase_timeline_data(tmp_path, rows)
    by_name = {name: (start, phase_end) for name, start, phase_end, _idx in phases}

    assert origin == pytest.approx(0.0)
    assert by_name["Mmproj"] == (pytest.approx(0.0), pytest.approx(1.0))
    assert by_name["ImagePrefill"] == (pytest.approx(1.0), pytest.approx(2.0))
    assert by_name["Decode"] == (pytest.approx(2.0), pytest.approx(3.0))
    assert end == pytest.approx(3.0)


def test_dynamic_kv_grow_separation_clips_vision_prefill_rows():
    rows = [
        _row("VisionPrefillMmproj", 1.30, 1.32),
        _row("DynamicKVGrow", 1.32, 1.46),
        _row("VisionPrefillImagePrefill", 1.32, 1.66),
        _row("VisionPrefillT_Prefill", 1.66, 1.72),
    ]

    separated = runner_cli._separate_dynamic_kv_grow_overlaps(rows)
    by_name = {row["row_type"]: row for row in separated}

    assert by_name["VisionPrefillImagePrefill"]["elapsed_s_start"] == "1.460000"
    assert by_name["VisionPrefillImagePrefill"]["total_ms"] == "200"


def test_hybrid_offline_vision_measured_encode_waits_for_start_gate():
    source = HYBRID_VISION_DUMP.read_text()

    session_idx = source.index("VisionEncoderSession session")
    ready_idx = source.index("write_text_file(FLAGS_ready_path")
    wait_idx = source.index("wait_for_file(")
    encode_idx = source.index("session.encode(image_paths)")

    assert session_idx < ready_idx < wait_idx < encode_idx


def test_hybrid_offline_decoder_warmup_finishes_before_ready_gate():
    source = HYBRID_DECODE.read_text()

    warmup_idx = source.index("warmup_mmproj_with_embedding(ctx, warmup_embedding)")
    ready_idx = source.index("write_text_file(custom.ready_path")
    measured_wait_idx = source.index("wait_for_file(custom.embedding_path")

    assert warmup_idx < ready_idx < measured_wait_idx
