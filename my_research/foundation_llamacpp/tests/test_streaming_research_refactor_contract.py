import subprocess
from pathlib import Path

from my_research.foundation_llamacpp.runner import media


ROOT = Path(__file__).resolve().parents[3]
PROMPT_FORMATS = ROOT / "my_research/foundation_llamacpp/runner/prompt_formats.py"
STREAMING_CPP = ROOT / "my_research/foundation_llamacpp/hybrid_bridge/hybrid_streaming_decode.cpp"
STREAMING_PROMPT_HEADER = ROOT / "my_research/foundation_llamacpp/hybrid_bridge/streaming_prompt_format.hpp"
STREAMING_POLICY_HEADER = ROOT / "my_research/foundation_llamacpp/hybrid_bridge/streaming_policy.hpp"
KV_REPOSITION = ROOT / "my_research/foundation_llamacpp/hybrid_bridge/kv_reposition.hpp"


def test_python_prompt_formatter_registry_preserves_internvl3_prompts():
    assert PROMPT_FORMATS.exists()

    formatter = media.get_prompt_formatter("internvl3")
    assert formatter.image_prompt("Describe.", n_images=1) == "<__media__>\nDescribe."
    assert (
        formatter.multi_image_prompt(2, "Compare.")
        == "Image-1: <__media__>\nImage-2: <__media__>\nCompare."
    )
    assert (
        formatter.video_prompt([1, 2], "What changed?")
        == "Frame1: <__media__>\nFrame2: <__media__><__media__>\nWhat changed?"
    )


def test_python_prompt_formatter_registry_exposes_qwen_extension_without_changing_default():
    assert media.normalize_prompt_format(None) == "internvl3"
    assert media.normalize_prompt_format("qwen2.5-vl") == "qwen2_5_vl"
    assert media.get_prompt_formatter("qwen2_5_vl").name == "qwen2_5_vl"
    assert media.internvl3_video_prompt([1], "Question?") == "Frame1: <__media__>\nQuestion?"


def test_streaming_cpp_uses_prompt_format_profile_helpers():
    source = STREAMING_CPP.read_text()
    header = STREAMING_PROMPT_HEADER.read_text()

    assert '#include "streaming_prompt_format.hpp"' in source
    assert "prompt_format_profile(args.prompt_format)" in source
    assert "build_stream_frame_prompt_line(profile" in source
    assert "build_stream_video_prompt_prefix(profile" in source
    assert "strip_stream_video_prompt_prefix(profile" in source

    assert "struct PromptFormatProfile" in header
    assert "qwen2_5_vl" in header
    assert "mrope" in header.lower()
    assert "profile.frame_prefix + std::to_string" in header


def test_streaming_prompt_header_compiles_as_extension_contract(tmp_path):
    probe = tmp_path / "prompt_profile_probe.cpp"
    probe.write_text(
        """
        #include "streaming_prompt_format.hpp"
        #include <string>
        #include <vector>

        struct Tile { std::string layout_image = "x.png"; std::string bin = "x.bin"; };
        struct Frame {
          int index = 2;
          double timestamp_s = 2.0;
          std::vector<Tile> tiles = {Tile{}, Tile{}};
        };

        int main() {
          using namespace streamingvlm::hybrid_bridge;
          PromptFormatProfile profile = prompt_format_profile("internvl3");
          Frame frame;
          std::string line = build_stream_frame_prompt_line(profile, frame);
          if (line != "Frame3: <__media__><__media__>\\n") {
            return 1;
          }
          std::vector<Frame> frames = {frame};
          std::string prefix = build_stream_video_prompt_prefix(profile, frames);
          if (strip_stream_video_prompt_prefix(profile, prefix + "Question?") != "Question?") {
            return 2;
          }
          if (prompt_format_profile("qwen2.5-vl").family != PromptFormatFamily::Qwen25VL) {
            return 3;
          }
          return 0;
        }
        """,
        encoding="utf-8",
    )
    probe_bin = tmp_path / "prompt_profile_probe"
    subprocess.run(
        [
            "g++",
            "-std=c++17",
            "-I",
            str(ROOT / "my_research/foundation_llamacpp/hybrid_bridge"),
            str(probe),
            "-o",
            str(probe_bin),
        ],
        check=True,
    )
    subprocess.run([str(probe_bin)], check=True)


def test_kv_reposition_declares_future_mrope_strategy_boundary():
    source = KV_REPOSITION.read_text()

    assert "enum class KvPositionEncodingKind" in source
    assert "Rope1D" in source
    assert "MRope" in source
    assert "struct KvRepositionStrategy" in source
    assert "requires_k_shift_rebuild" in source


def test_streaming_policy_header_owns_frame_selection_modes():
    source = STREAMING_CPP.read_text()
    header = STREAMING_POLICY_HEADER.read_text()

    assert '#include "streaming_policy.hpp"' in source
    assert "StreamingPolicyConfig" in header
    assert "select_prompt_frames(" in source
    assert "policy," in source
    assert 'stream_mode == "on_demand"' in header
    assert 'stream_mode == "vision_prefill"' in header
    assert 'stream_mode != "sliding_window"' in header
    assert "evenly_limit_items" in header


def test_streaming_policy_header_compiles_as_extension_contract(tmp_path):
    probe = tmp_path / "streaming_policy_probe.cpp"
    probe.write_text(
        """
        #include "streaming_policy.hpp"
        #include <vector>

        struct Frame {
          int index = 0;
          double timestamp_s = 0.0;
        };

        int main() {
          using namespace streamingvlm::hybrid_bridge;
          std::vector<Frame> frames = {{0, 0.0}, {1, 1.0}, {2, 2.0}, {3, 3.0}};
          StreamingPolicyConfig policy;
          policy.stream_mode = "sliding_window";
          policy.window_sec = 1.5;
          policy.window_max_frames = 2;
          auto selected = select_prompt_frames(policy, frames, frames.back(), 3.0);
          if (selected.size() != 2 || selected[0].index != 2 || selected[1].index != 3) {
            return 1;
          }
          policy.stream_mode = "vision_prefill";
          selected = select_prompt_frames(policy, frames, frames.back(), 3.0);
          if (selected.size() != 4) {
            return 2;
          }
          policy.stream_mode = "on_demand";
          selected = select_prompt_frames(policy, frames, frames.back(), 3.0);
          if (selected.size() != 1 || selected[0].index != 3) {
            return 3;
          }
          return 0;
        }
        """,
        encoding="utf-8",
    )
    probe_bin = tmp_path / "streaming_policy_probe"
    subprocess.run(
        [
            "g++",
            "-std=c++17",
            "-I",
            str(ROOT / "my_research/foundation_llamacpp/hybrid_bridge"),
            str(probe),
            "-o",
            str(probe_bin),
        ],
        check=True,
    )
    subprocess.run([str(probe_bin)], check=True)
