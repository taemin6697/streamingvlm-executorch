import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
HEADER = ROOT / "my_research/foundation_llamacpp/hybrid_bridge/kv_reposition.hpp"
CMAKE = ROOT / "my_research/foundation_llamacpp/hybrid_bridge/CMakeLists.txt"
ARCHIVE = ROOT / "my_research/foundation_llamacpp/docs/archive/kv_rope_reposition_for_video_compression.md"
README = ROOT / "my_research/foundation_llamacpp/docs/README.md"
STRUCTURE = ROOT / "my_research/foundation_llamacpp/docs/project_structure.md"


def test_kv_reposition_header_defines_tail_compaction_contract():
    source = HEADER.read_text()

    assert "struct KvTokenRange" in source
    assert "struct KvTailCompactionPlan" in source
    assert "struct KvTailInsertionPlan" in source
    assert "struct KvTailParkingPlan" in source
    assert "build_tail_compaction_plan" in source
    assert "build_tail_insertion_plan" in source
    assert "build_tail_parking_plan" in source
    assert "apply_tail_compaction_plan" in source
    assert "apply_tail_insertion_plan" in source
    assert "apply_tail_parking_plan" in source
    assert "restore_parked_tail_after_insert" in source
    assert "compacted_position_after" in source
    assert "inserted_position_after" in source
    assert "llama_memory_seq_rm" in source
    assert "llama_memory_seq_cp" in source
    assert "llama_memory_seq_add" in source
    assert "llama_memory_seq_div" not in source


def test_kv_reposition_probe_is_buildable_from_hybrid_bridge_cmake():
    source = CMAKE.read_text()

    assert "add_executable(\n    kv_reposition_probe" in source
    assert "kv_reposition_probe.cpp" in source
    assert "target_link_libraries(kv_reposition_probe PRIVATE llama Threads::Threads)" in source


def test_kv_reposition_header_compiles_as_standalone_contract(tmp_path):
    probe = tmp_path / "kv_reposition_probe.cpp"
    probe.write_text(
        """
        #include "kv_reposition.hpp"
        #include <string>

        int main() {
          using namespace streamingvlm::hybrid_bridge;
          KvTailCompactionPlan plan;
          std::string error;
          const bool ok = build_tail_compaction_plan(KvTokenRange{128, 384}, 1024, &plan, &error);
          if (!ok || plan.shift != -256 || plan.tail_begin != 384 || plan.tail_end != 1024) {
            return 1;
          }
          if (compacted_position_after(KvTokenRange{128, 384}, 384) != 128) {
            return 2;
          }
          if (compacted_position_after(KvTokenRange{128, 384}, 256) != -1) {
            return 3;
          }
          KvTailInsertionPlan insert_plan;
          const bool insert_ok = build_tail_insertion_plan(384, 256, 1024, &insert_plan, &error);
          if (!insert_ok || insert_plan.shift != 256 || insert_plan.tail_begin != 384 ||
              insert_plan.tail_end != 1024 || insert_plan.expanded_sequence_end != 1280) {
            return 4;
          }
          if (inserted_position_after(384, 256, 1024, 384) != 640) {
            return 5;
          }
          if (inserted_position_after(384, 256, 1024, 128) != 128) {
            return 6;
          }
          KvTailParkingPlan park_plan;
          const bool park_ok = build_tail_parking_plan(384, 1024, 0, 1, &park_plan, &error);
          if (!park_ok || park_plan.tail_begin != 384 || park_plan.tail_end != 1024 ||
              park_plan.main_seq_id != 0 || park_plan.scratch_seq_id != 1) {
            return 7;
          }
          return 0;
        }
        """,
        encoding="utf-8",
    )

    probe_bin = tmp_path / "kv_reposition_probe"
    subprocess.run(
        [
            "g++",
            "-std=c++17",
            "-I",
            str(ROOT / "my_research/foundation_llamacpp/hybrid_bridge"),
            "-I",
            str(ROOT / "llama.cpp/include"),
            "-I",
            str(ROOT / "llama.cpp/ggml/include"),
            str(probe),
            "-o",
            str(probe_bin),
        ],
        check=True,
    )
    subprocess.run(
        [
            str(probe_bin),
        ],
        check=True,
    )


def test_rope_reposition_design_is_documented_for_future_video_compression():
    archive = ARCHIVE.read_text()
    readme = README.read_text()
    structure = STRUCTURE.read_text()

    assert "kv_reposition.hpp" in archive
    assert "llama_memory_seq_rm" in archive
    assert "llama_memory_seq_cp" in archive
    assert "llama_memory_seq_add" in archive
    assert "cached K" in archive
    assert "V cache" in archive
    assert "M-RoPE" in archive

    assert "kv_rope_reposition_for_video_compression.md" in readme
    assert "kv_reposition.hpp" in structure


def test_streaming_vision_prefill_can_compact_cached_frame_spans():
    source = (ROOT / "my_research/foundation_llamacpp/hybrid_bridge/hybrid_streaming_decode.cpp").read_text()
    runner = (ROOT / "my_research/foundation_llamacpp/runner/cli.py").read_text()

    assert "kv_reposition_keep_latest_frames" in source
    assert "--kv-reposition-keep-latest-frames" in source
    assert "struct VisionKvSpan" in source
    assert "frame_kv_spans" in source
    assert "compact_vision_prefill_cache_frames" in source
    assert "build_tail_compaction_plan" in source
    assert "apply_tail_compaction_plan" in source
    assert "KVRepositionCompact" in source
    assert "kv_reposition_compactions" in source
    assert "args.latest_frame_only\n                           ? std::vector<FrameRecord>{current_frame}" in source
    assert "job.frames = {latest_frame};" in source

    assert "--kv-reposition-keep-latest-frames" in runner
