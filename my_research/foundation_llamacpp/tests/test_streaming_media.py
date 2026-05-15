import pytest

from my_research.foundation_llamacpp.runner.media import (
    MEDIA_MARKER,
    build_streaming_video_prompt,
    normalize_stream_mode,
    select_recent_window_frames,
)


def test_normalize_stream_mode_keeps_single_buffer_alias():
    assert normalize_stream_mode(None, single_buffer=True) == "on_demand"
    assert normalize_stream_mode(None, single_buffer=False) == "on_demand"
    assert normalize_stream_mode("on-demand", single_buffer=False) == "on_demand"
    assert normalize_stream_mode("single-buffer", single_buffer=False) == "on_demand"
    assert normalize_stream_mode("sliding-window", single_buffer=False) == "sliding_window"
    assert normalize_stream_mode("vision-prefill", single_buffer=False) == "vision_prefill"


def test_normalize_stream_mode_rejects_unknown_mode():
    with pytest.raises(ValueError, match="unsupported stream mode"):
        normalize_stream_mode("rolling-cache", single_buffer=False)


def test_select_recent_window_frames_filters_by_prompt_time_and_window():
    frames = [
        {"stream_frame": 0, "timestamp_s": 0.0},
        {"stream_frame": 1, "timestamp_s": 1.0},
        {"stream_frame": 2, "timestamp_s": 2.0},
        {"stream_frame": 3, "timestamp_s": 3.0},
        {"stream_frame": 4, "timestamp_s": 4.0},
    ]

    selected = select_recent_window_frames(
        frames,
        prompt_time_s=3.25,
        window_sec=2.0,
        window_max_frames=8,
    )

    assert [frame["stream_frame"] for frame in selected] == [2, 3]


def test_select_recent_window_frames_evenly_limits_long_window():
    frames = [{"stream_frame": idx, "timestamp_s": float(idx)} for idx in range(10)]

    selected = select_recent_window_frames(
        frames,
        prompt_time_s=9.0,
        window_sec=9.0,
        window_max_frames=4,
    )

    assert [frame["stream_frame"] for frame in selected] == [0, 3, 6, 9]


def test_build_streaming_video_prompt_uses_internvl_frame_format():
    frames = [
        {"num_patches": 1},
        {"num_patches": 2},
    ]

    prompt = build_streaming_video_prompt(frames, "What changed?")

    assert prompt == (
        f"Frame1: {MEDIA_MARKER}\n"
        f"Frame2: {MEDIA_MARKER}{MEDIA_MARKER}\n"
        "What changed?"
    )
