from __future__ import annotations

import os
import subprocess
import tempfile
import shlex
from pathlib import Path

import numpy as np
from my_research.foundation.manifest import (
    FoundationManifest,
    load_manifest,
)


def _run(cmd: list[str]) -> int:
    return subprocess.run(cmd, check=True).returncode


def _artifact_root(manifest: FoundationManifest) -> Path:
    return Path(manifest.paths["artifact_root"]).resolve()


def _local_log_dir(manifest: FoundationManifest) -> Path:
    foundation_dir = Path(__file__).resolve().parents[1]
    artifact_name = _artifact_root(manifest).name
    output_dir = foundation_dir / "results" / "log" / manifest.backend / artifact_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _local_output_path(manifest: FoundationManifest) -> Path:
    return _local_log_dir(manifest) / "foundation_output.txt"


def _remote_cache_root(manifest: FoundationManifest) -> str:
    artifact_name = _artifact_root(manifest).name
    return f"/data/local/tmp/foundation_runner/{artifact_name}"


def _adb_base(serial: str | None) -> list[str]:
    cmd = ["adb"]
    if serial:
        cmd.extend(["-s", serial])
    return cmd


def _remote_file_exists(adb: list[str], remote_path: str) -> bool:
    result = subprocess.run(
        adb + ["shell", f"test -f {shlex.quote(remote_path)}"],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def _remote_file_size(adb: list[str], remote_path: str) -> int | None:
    result = subprocess.run(
        adb + ["shell", f"stat -c %s {shlex.quote(remote_path)}"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    if result.returncode != 0:
        return None
    try:
        return int(result.stdout.strip())
    except ValueError:
        return None


def _push_file_cached(
    adb: list[str],
    local_path: str | Path,
    remote_path: str,
    *,
    force_push: bool,
) -> None:
    local_path = Path(local_path)
    if not force_push:
        remote_size = _remote_file_size(adb, remote_path)
        local_size = local_path.stat().st_size
        if remote_size == local_size:
            print(f"[foundation] device cache hit, skip push: {remote_path}")
            return
        if remote_size is not None:
            print(
                "[foundation] device cache stale, re-push: "
                f"{remote_path} ({remote_size} -> {local_size} bytes)"
            )
    _run(adb + ["push", str(local_path), remote_path])


def _pull_if_exists(adb: list[str], remote_path: str, local_path: Path) -> bool:
    if not _remote_file_exists(adb, remote_path):
        return False
    local_path.parent.mkdir(parents=True, exist_ok=True)
    _run(adb + ["pull", remote_path, str(local_path)])
    return True


def _finalize_run_logs(
    adb: list[str],
    manifest: FoundationManifest,
    remote_root: str,
    *,
    save_log: bool,
) -> int:
    log_dir = _local_log_dir(manifest)
    rc = _run(
        adb
        + [
            "pull",
            f"{remote_root}/foundation_output.txt",
            str(log_dir / "foundation_output.txt"),
        ]
    )
    if save_log:
        pulled = False
        for name in (
            "foundation_proc.csv",
            "android_memory_timeline.csv",
            "vision_output_stats.csv",
            "vision_output_0000_f32.bin",
        ):
            pulled = (
                _pull_if_exists(adb, f"{remote_root}/{name}", log_dir / name) or pulled
            )
        if pulled:
            try:
                from my_research.foundation.host.memory_plot import (
                    generate_memory_timeline_plot,
                )

                plot_path = generate_memory_timeline_plot(log_dir)
                if plot_path:
                    print(f"[foundation] memory plot: {plot_path}")
            except Exception as exc:
                print(f"[foundation] warning: failed to generate memory plot: {exc}")
    return rc


def _normalize_image_to_bin(image, output_path: Path, image_size: int = 448) -> None:
    from PIL import Image

    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    image = image.convert("RGB").resize((image_size, image_size))
    arr = np.asarray(image).astype("float32") / 255.0
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    arr = np.transpose(arr, (2, 0, 1))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    arr.astype("float32").tofile(output_path)


def _extract_image(image: str, output_dir: Path, variant: str) -> int:
    from transformers.image_utils import load_image

    del variant
    _normalize_image_to_bin(load_image(image), output_dir / "frame_0000.bin")
    return 1


def _extract_frames(video: str, fps: float, output_dir: Path, variant: str) -> int:
    del variant
    try:
        import cv2
    except ImportError as exc:
        raise SystemExit("Video input requires opencv-python (`cv2`).") from exc

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"Failed to open video: {video}")
    source_fps = cap.get(cv2.CAP_PROP_FPS) or fps
    step = max(int(round(source_fps / fps)), 1)
    frame_idx = 0
    saved = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % step == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            _normalize_image_to_bin(rgb, output_dir / f"frame_{saved:04d}.bin")
            saved += 1
        frame_idx += 1
    cap.release()
    if saved == 0:
        raise SystemExit(f"No frames extracted from video: {video}")
    return saved


class _AdbWorkspace:
    DEVICE_WORKSPACE_ROOT = "/data/local/tmp/foundation_runner"

    def __init__(
        self,
        *,
        build_path: str,
        device_id: str,
        qnn_sdk: str,
        soc_model: str,
        device_workspace: str,
        force_push: bool,
    ) -> None:
        self.build_path = Path(build_path)
        self.device_id = device_id
        self.qnn_sdk = Path(qnn_sdk)
        self.soc_model = soc_model
        self.device_workspace = device_workspace
        self.force_push = force_push
        self.adb = _adb_base(device_id)

    def _htp_arch(self) -> str:
        mapping = {
            "SM8750": "79",
            "SM8650": "75",
            "SM8550": "73",
            "SM8450": "69",
            "SM8350": "68",
        }
        return mapping.get(self.soc_model, "73")

    def setup_workspace(self) -> None:
        if self.force_push:
            _run(self.adb + ["shell", "rm", "-rf", self.device_workspace])
        _run(self.adb + ["shell", "mkdir", "-p", self.device_workspace])

    def push_file(self, path: str, *, force_push: bool | None = None) -> None:
        local_path = Path(path)
        remote_path = f"{self.device_workspace}/{local_path.name}"
        _push_file_cached(
            self.adb,
            local_path,
            remote_path,
            force_push=self.force_push if force_push is None else force_push,
        )

    def push_dir_files(
        self,
        directory: str,
        pattern: str,
        *,
        force_push: bool | None = None,
    ) -> None:
        for path in sorted(Path(directory).glob(pattern)):
            self.push_file(str(path), force_push=force_push)

    def push_qnn_libs(self) -> None:
        qnn_lib_dir = self.qnn_sdk / "lib" / "aarch64-android"
        if qnn_lib_dir.exists():
            for lib in sorted(qnn_lib_dir.glob("libQnn*.so")):
                self.push_file(str(lib))
        arch = self._htp_arch()
        htp_skel = (
            self.qnn_sdk
            / "lib"
            / f"hexagon-v{arch}"
            / "unsigned"
            / f"libQnnHtpV{arch}Skel.so"
        )
        if htp_skel.exists():
            self.push_file(str(htp_skel))
        else:
            print(f"[foundation] warning: missing QNN HTP skel library: {htp_skel}")
        backend_lib = self.build_path / "backends" / "qualcomm" / "libqnn_executorch_backend.so"
        if backend_lib.exists():
            self.push_file(str(backend_lib))

    def execute(self, command: str) -> str:
        full_cmd = f"cd {self.device_workspace} && {command}"
        return subprocess.check_output(
            self.adb + ["shell", full_cmd],
            text=True,
            stderr=subprocess.STDOUT,
        )


def _run_unified_xnnpack(
    manifest: FoundationManifest,
    manifest_path: Path,
    *,
    runner_binary: str,
    device: str | None,
    image: str | None,
    video: str | None,
    questions: list[str],
    timestamps: list[float] | None,
    seq_len: int | None,
    temperature: float | None,
    eval_mode: int = 0,
    save_log: bool = False,
    vision_only: bool = False,
    force_push: bool = False,
) -> int:
    adb = _adb_base(device)
    remote_root = _remote_cache_root(manifest)
    remote_frames = f"{remote_root}/frames"
    remote_runner = f"{remote_root}/xnnpack_qnn_runner"
    text_only = image is None and video is None
    decoder_input_mode = (
        manifest.metadata.get("decoder_input_mode")
        or manifest.export.get("decoder_input_mode")
        or "embeddings"
    )

    with tempfile.TemporaryDirectory(prefix="foundation_xnnpack_") as tmpdir:
        tmpdir = Path(tmpdir)
        frame_dir = tmpdir / "frames"
        frame_dir.mkdir(parents=True, exist_ok=True)

        if image:
            frame_count = _extract_image(image, frame_dir, manifest.variant)
        elif video:
            frame_count = _extract_frames(video, 1.0, frame_dir, manifest.variant)
        else:
            frame_count = 0

        device_manifest = manifest.resolve_paths(manifest_path.parent)
        rel_paths = {}
        for key, value in device_manifest.paths.items():
            if key == "artifact_root":
                rel_paths[key] = "."
            elif value:
                rel_paths[key] = Path(value).name
        device_manifest.paths = rel_paths

        if force_push:
            _run(adb + ["shell", "rm", "-rf", remote_root])
        _run(adb + ["shell", "mkdir", "-p", remote_root])
        _run(adb + ["push", str(runner_binary), remote_runner])
        # xnnpack_qnn_runner links to libqnn_executorch_backend.so; push it for load
        qnn_lib = Path(runner_binary).resolve().parent.parent / "backends/qualcomm/libqnn_executorch_backend.so"
        if qnn_lib.exists():
            _push_file_cached(
                adb,
                qnn_lib,
                f"{remote_root}/{qnn_lib.name}",
                force_push=force_push,
            )

        push_keys = (
            ("vision_encoder_pte",)
            if vision_only
            else (
                "vision_encoder_pte",
                "text_embedding_pte",
                "text_decoder_pte",
                "tokenizer_path",
            )
        )
        for key in push_keys:
            path = manifest.paths.get(key)
            if path:
                _push_file_cached(
                    adb,
                    path,
                    f"{remote_root}/{Path(path).name}",
                    force_push=force_push,
                )

        if not text_only:
            # Inputs can change between runs, so refresh only the frame directory.
            _run(adb + ["shell", "rm", "-rf", remote_frames])
            _run(adb + ["shell", "mkdir", "-p", remote_frames])
            # adb push dir dest -> dest/dir/ 생성. 내용만 푸시하려면 dir/. 사용
            _run(adb + ["push", str(frame_dir) + "/.", remote_frames])
        _run(adb + ["shell", "chmod", "+x", remote_runner])

        encoder_path = device_manifest.paths.get("vision_encoder_pte", "encoder.pte")
        embedding_path = device_manifest.paths.get("text_embedding_pte", "embedding.pte")
        decoder_path = device_manifest.paths.get("text_decoder_pte", "decoder.pte")
        tokenizer_path = device_manifest.paths.get("tokenizer_path", "tokenizer.bin")

        args = [
            f"--backend={manifest.backend}",
            f"--encoder_path={encoder_path}",
            f"--embedding_path={embedding_path}",
            f"--decoder_path={decoder_path}",
            f"--tokenizer_path={tokenizer_path}",
            f"--seq_len={seq_len or 128}",
            f"--temperature={temperature if temperature is not None else 0.0}",
            f"--eval_mode={eval_mode}",
            f"--output_path=foundation_output.txt",
        ]
        if not text_only:
            args.append(f"--image_path={remote_frames}")
        args.append(f"--decoder_input_mode={decoder_input_mode}")
        if vision_only:
            args.append("--vision_only")
        for q in questions:
            args.extend(["--prompt", shlex.quote(q)])
        if save_log:
            args.append("--save_log")
        shell_cmd = f"cd {remote_root} && export LD_LIBRARY_PATH=. && ./xnnpack_qnn_runner " + " ".join(args)
        _run(adb + ["shell", shell_cmd])
        return _finalize_run_logs(
            adb,
            manifest,
            remote_root,
            save_log=save_log,
        )


def _run_unified_qnn(
    manifest: FoundationManifest,
    manifest_path: Path,
    *,
    runner_binary: str,
    build_path: str,
    device: str,
    model: str,
    image: str | None,
    video: str | None,
    questions: list[str],
    timestamps: list[float] | None,
    seq_len: int | None,
    temperature: float | None,
    eval_mode: int = 0,
    save_log: bool = False,
    force_push: bool = False,
) -> int:
    qnn_sdk = os.environ.get("QNN_SDK_ROOT", "")
    if not qnn_sdk:
        raise SystemExit("QNN 실행에는 환경변수 QNN_SDK_ROOT 가 필요합니다.")

    with tempfile.TemporaryDirectory(prefix="foundation_qnn_") as tmpdir:
        tmpdir = Path(tmpdir)
        frame_dir = tmpdir / "frames"
        frame_dir.mkdir(parents=True, exist_ok=True)

        if image:
            frame_count = _extract_image(image, frame_dir, manifest.variant)
        elif video:
            frame_count = _extract_frames(video, 1.0, frame_dir, manifest.variant)
        else:
            frame_count = 0

        device_manifest = manifest.resolve_paths(manifest_path.parent)
        rel_paths = {}
        for key, value in device_manifest.paths.items():
            if key == "artifact_root":
                rel_paths[key] = "."
            elif value:
                rel_paths[key] = Path(value).name
        device_manifest.paths = rel_paths

        adb = _AdbWorkspace(
            build_path=str(build_path),
            device_id=device,
            qnn_sdk=qnn_sdk,
            soc_model=model,
            device_workspace=_remote_cache_root(manifest),
            force_push=force_push,
        )
        adb.setup_workspace()
        adb.push_qnn_libs()
        _run(adb.adb + ["push", str(runner_binary), adb.device_workspace])
        for key in (
            "vision_encoder_pte",
            "text_embedding_pte",
            "text_decoder_pte",
            "tokenizer_path",
        ):
            path = manifest.paths.get(key)
            if path:
                adb.push_file(str(path))
        if frame_count > 0:
            # Inputs can change between runs even if the model cache is reused.
            adb.push_dir_files(str(frame_dir), "*.bin", force_push=True)

        encoder_path = device_manifest.paths.get("vision_encoder_pte", "encoder.pte")
        embedding_path = device_manifest.paths.get("text_embedding_pte", "embedding.pte")
        decoder_path = device_manifest.paths.get("text_decoder_pte", "decoder.pte")
        tokenizer_path = device_manifest.paths.get("tokenizer_path", "tokenizer.bin")
        image_path = "frame_0000.bin" if frame_count > 0 else ""
        if not image_path:
            raise SystemExit("QNN foundation 실행에는 --image 또는 --video 가 필요합니다.")

        cmd = [
            "chmod +x ./xnnpack_qnn_runner &&",
            "export LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. &&",
            "./xnnpack_qnn_runner",
            "--backend=qnn",
            f"--encoder_path={encoder_path}",
            f"--embedding_path={embedding_path}",
            f"--decoder_path={decoder_path}",
            f"--tokenizer_path={tokenizer_path}",
            f"--image_path={image_path}",
            f"--seq_len={seq_len or manifest.export.get('max_seq_len', 1024)}",
            f"--temperature={temperature if temperature is not None else 0.0}",
            f"--eval_mode={eval_mode}",
            "--output_path=foundation_output.txt",
        ]
        for q in questions:
            cmd.extend(["--prompt", shlex.quote(q)])
        if save_log:
            cmd.append("--save_log")
        runner_out = adb.execute(" ".join(cmd))
        if runner_out:
            print(runner_out)
        try:
            return _finalize_run_logs(
                _adb_base(device),
                manifest,
                adb.device_workspace,
                save_log=save_log,
            )
        except subprocess.CalledProcessError:
            print(
                "\n[foundation] 러너가 foundation_output.txt 를 생성하지 못했습니다. "
                "위 디바이스 출력을 확인하세요."
            )
            raise


def run_with_manifest(
    manifest_path: Path,
    *,
    build_path: str | None = None,
    device: str | None = None,
    model: str | None = None,
    image: str | None = None,
    video: str | None = None,
    questions: list[str] | None = None,
    timestamps: list[float] | None = None,
    seq_len: int | None = None,
    temperature: float | None = None,
    eval_mode: int = 0,
    save_log: bool = False,
    stream: bool = False,
    runner_binary: str | None = None,
    force_push: bool = False,
    vision_only: bool = False,
) -> int:
    manifest = load_manifest(Path(manifest_path))
    artifact_root = _artifact_root(manifest)
    questions = questions or ["Describe this image."]
    if eval_mode is None:
        eval_mode = 1 if manifest.backend == "qnn" else 0

    if manifest.backend == "qnn":
        if not build_path or not device or not model:
            raise SystemExit("QNN 실행에는 --build_path, --device, --model 이 필요합니다.")
        if not runner_binary or Path(runner_binary).name != "xnnpack_qnn_runner":
            raise SystemExit(
                "QNN foundation 실행에는 --runner_binary xnnpack_qnn_runner 가 필요합니다."
            )
        return _run_unified_qnn(
            manifest,
            Path(manifest_path),
            runner_binary=runner_binary,
            build_path=build_path,
            device=device,
            model=model,
            image=image,
            video=video,
            questions=questions,
            timestamps=timestamps,
            seq_len=seq_len,
            temperature=temperature,
            eval_mode=eval_mode,
            save_log=save_log,
            force_push=force_push,
        )

    if manifest.backend in {"xnnpack", "vulkan"}:
        if not runner_binary:
            raise SystemExit(
                f"{manifest.backend} foundation 실행에는 --runner_binary xnnpack_qnn_runner 가 필요합니다."
            )
        if Path(runner_binary).name != "xnnpack_qnn_runner":
            raise SystemExit(
                f"{manifest.backend} foundation 실행은 xnnpack_qnn_runner 만 지원합니다."
            )
        return _run_unified_xnnpack(
            manifest,
            Path(manifest_path),
            runner_binary=runner_binary,
            device=device,
            image=image,
            video=video,
            questions=questions,
            timestamps=timestamps,
            seq_len=seq_len,
            temperature=temperature,
            eval_mode=eval_mode,
            save_log=save_log,
            force_push=force_push,
            vision_only=vision_only,
        )

    raise SystemExit(f"지원하지 않는 backend: {manifest.backend}")
