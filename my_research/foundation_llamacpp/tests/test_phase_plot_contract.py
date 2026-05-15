import pytest

from my_research.foundation_llamacpp.runner import cli as runner_cli


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
