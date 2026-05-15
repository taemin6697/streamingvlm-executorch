import argparse
import json

from PIL import Image

from my_research.foundation_llamacpp.runner import cli as runner_cli
from my_research.foundation_llamacpp.runner.media import MEDIA_MARKER, prepare_media


def test_organize_result_artifacts_moves_run_files_into_three_folders(tmp_path):
    result_dir = tmp_path / "InternVL3-1B-Instruct-Q8_0_hybrid_ctx_4096_kv16"
    result_dir.mkdir()
    for name, text in {
        "foundation_proc.csv": "row_type,total_ms\nDecode,1\n",
        "phase_timeline.png": "png-bytes",
        "foundation_output.txt": "assistant output",
        "media_manifest.json": "{}\n",
        "vision_embedding.svlmemb": "raw embedding",
    }.items():
        (result_dir / name).write_text(text, encoding="utf-8")

    runner_cli._organize_result_artifacts(result_dir)

    assert (result_dir / "csv" / "foundation_proc.csv").exists()
    assert (result_dir / "png" / "phase_timeline.png").exists()
    assert (result_dir / "txt_json" / "foundation_output.txt").exists()
    assert (result_dir / "txt_json" / "media_manifest.json").exists()
    assert (result_dir / "txt_json" / "vision_embedding.svlmemb").exists()
    assert sorted(path.name for path in result_dir.iterdir()) == ["csv", "png", "txt_json"]


def test_organize_result_artifacts_is_idempotent_and_overwrites_old_grouped_file(tmp_path):
    result_dir = tmp_path / "run"
    (result_dir / "csv").mkdir(parents=True)
    (result_dir / "csv" / "foundation_summary.csv").write_text("old\n", encoding="utf-8")
    (result_dir / "foundation_summary.csv").write_text("new\n", encoding="utf-8")

    runner_cli._organize_result_artifacts(result_dir)
    runner_cli._organize_result_artifacts(result_dir)

    assert (result_dir / "csv" / "foundation_summary.csv").read_text(encoding="utf-8") == "new\n"
    assert sorted(path.name for path in result_dir.iterdir()) == ["csv", "png", "txt_json"]


def test_clear_result_artifact_dirs_removes_grouped_and_root_files(tmp_path):
    result_dir = tmp_path / "run"
    result_dir.mkdir()
    (result_dir / "stale_root.txt").write_text("old\n", encoding="utf-8")
    (result_dir / "csv").mkdir()
    (result_dir / "csv" / "stale.csv").write_text("old\n", encoding="utf-8")

    runner_cli._clear_result_artifact_dirs(result_dir)

    assert not (result_dir / "stale_root.txt").exists()
    assert not (result_dir / "csv" / "stale.csv").exists()


def test_prepare_multi_image_media_uses_internvl_image_prefixes(tmp_path):
    image_a = tmp_path / "a.jpg"
    image_b = tmp_path / "b.jpg"
    Image.new("RGB", (32, 32), color=(255, 0, 0)).save(image_a)
    Image.new("RGB", (32, 32), color=(0, 255, 0)).save(image_b)
    args = argparse.Namespace(
        images=[image_a, image_b],
        image=None,
        video=None,
        streaming_video=None,
        prompt="Compare the two images.",
    )

    media = prepare_media(args, tmp_path / "prepared")
    manifest = json.loads(media.metadata_path.read_text(encoding="utf-8"))

    assert media.source_kind == "multi_image"
    assert media.prompt == (
        f"Image-1: {MEDIA_MARKER}\n"
        f"Image-2: {MEDIA_MARKER}\n"
        "Compare the two images."
    )
    assert manifest["source_kind"] == "multi_image"
    assert manifest["num_patches_list"] == [1, 1]
    assert len(manifest["layout_images"]) == 2


def test_result_model_name_separates_image_multi_image_and_video_runs(tmp_path):
    model = tmp_path / "InternVL3-1B-Instruct-Q8_0.gguf"

    image_name = runner_cli._result_model_name(model, "hybrid", 4096, media_mode="image")
    multi_name = runner_cli._result_model_name(model, "hybrid", 4096, media_mode="multi_image")
    video_name = runner_cli._result_model_name(model, "hybrid", 4096, media_mode="video_file")

    assert "_image_" in image_name
    assert "_multi_image_" in multi_name
    assert "_video_" in video_name
    assert len({image_name, multi_name, video_name}) == 3


def test_artifact_layout_smoke_script_runs_vision_prefill_with_dynamic_kv():
    script = (
        runner_cli.FOUNDATION_LLAMA
        / "scripts"
        / "run_artifact_layout_1b_q8.sh"
    ).read_text(encoding="utf-8")

    assert script.count("--stream-mode vision-prefill") == 2
    assert "--processor hybrid" in script
    assert "--vision \"$VISION\"" in script
    assert "--dynamic-kv-cache" in script
    assert "--kv-init-size 512" in script
    assert "--kv-grow-step 512" in script
    assert "_streaming_vision_prefill_kv16\"" in script
    assert "_streaming_vision_prefill_kv16_dynamic" in script
