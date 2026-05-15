#!/usr/bin/env python3
"""
InternVL3-1B (HF) video QA — mirrors OpenGVLab README video path (decord + FrameN: <image>).

Default video: repo sample `surveil_8.mp4`.
Requires: CUDA GPU recommended; transformers, torch, torchvision, pillow, numpy, decord.

Important: use transformers 4.x only (e.g. `pip install 'transformers>=4.45,<5'`).
transformers v5/rc breaks OpenGVLab InternVL remote code (`all_tied_weights_keys`).

Run outputs go to `my_research/test_python/results/<video_stem>_<UTC>/`: full text `*.txt`,
token rows `*.tokens.tsv`, and `foundation_token_io.txt` (layout similar to llama.cpp hybrid/opencl log).
Synthetic vision rows use `-1` and `<VISION_KV_SLOT>`; count = sum(num_patches_list)*slots_per_patch.
Slot labels use per-frame denominator (e.g. `1/256`…`256/256` when one patch × 256), matching mtmd per-image chunks.
"""

from __future__ import annotations

import argparse
import importlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
import transformers
from transformers import AutoModel, AutoTokenizer


def _require_transformers_v4() -> None:
    major_str = transformers.__version__.split(".")[0]
    try:
        major = int(major_str)
    except ValueError:
        return
    if major >= 5:
        sys.exit(
            f"Incompatible transformers {transformers.__version__}. "
            "OpenGVLab InternVL (trust_remote_code) expects transformers 4.x.\n"
            "Fix: pip install 'transformers>=4.45,<5'"
        )

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VIDEO = REPO_ROOT / "my_research/foundation_llamacpp/sample_images/surveil_8.mp4"
RESULTS_PARENT = REPO_ROOT / "my_research/test_python/results"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size: int):
    return T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(
    image: Image.Image,
    min_num: int = 1,
    max_num: int = 12,
    image_size: int = 448,
    use_thumbnail: bool = False,
):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array(
        [int(start_idx + (seg_size / 2) + np.round(seg_size * idx)) for idx in range(num_segments)]
    )
    return frame_indices


def load_video(video_path: Path, bound=None, input_size=448, max_num=1, num_segments=32):
    vr = VideoReader(str(video_path), ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())

    pixel_values_list, num_patches_list = [], []
    transform = build_transform(input_size=input_size)
    frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
    for frame_index in frame_indices:
        img = Image.fromarray(vr[int(frame_index)].asnumpy()).convert("RGB")
        tiles = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = torch.stack([transform(tile) for tile in tiles])
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)
    pixel_values = torch.cat(pixel_values_list, dim=0)
    return pixel_values, num_patches_list


def write_token_tsv(path: Path, tokenizer, text: str, *, add_special_tokens: bool = False) -> int:
    """One row per tokenizer output token: idx, id, subword_piece, decode([id])."""
    ids = tokenizer.encode(text, add_special_tokens=add_special_tokens)
    rows = ["# idx\ttoken_id\tsubword_piece\tdecode_single_id"]
    for i, tid in enumerate(ids):
        subtokens = tokenizer.convert_ids_to_tokens([tid])
        sub = subtokens[0] if subtokens else ""
        frag = tokenizer.decode([tid])
        safe_sub = sub.replace("\t", "\\t").replace("\n", "\\n").replace("\r", "\\r")
        safe_frag = frag.replace("\t", "\\t").replace("\n", "\\n").replace("\r", "\\r")
        rows.append(f"{i}\t{tid}\t{safe_sub}\t{safe_frag}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return len(ids)


def write_plain(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def collapse_img_context_runs(query: str, marker: str = "<IMG_CONTEXT>") -> str:
    """Collapse `<IMG_CONTEXT><IMG_CONTEXT>...` for readable logs."""
    esc = re.escape(marker)
    pat = re.compile(f"(?:{esc})+")

    def repl(m: re.Match[str]) -> str:
        n = len(m.group(0)) // len(marker)
        return f"{marker}×{n}"

    return pat.sub(repl, query)


def print_internvl_preflight_after_vision_processor(
    model,
    tokenizer,
    *,
    question: str,
    num_patches_list: list[int],
    pixel_values: torch.Tensor,
    history: list | None = None,
    img_start: str = "<img>",
    img_end: str = "</img>",
    img_ctx: str = "<IMG_CONTEXT>",
) -> None:
    """
    Same string/tensor path as InternVLChatModel.chat → tokenizer(query), then extract_feature(pixel_values).

    Prints (1) expanded prompt with IMG_CONTEXT runs collapsed, (2) input_ids / pieces with IMG_CONTEXT
    ids marked, (3) ViT+MLP (`extract_feature`) tensor shape (what gets pasted into those positions in generate).
    """
    parent_pkg = model.__class__.__module__.rsplit(".", 1)[0]
    conv = importlib.import_module(parent_pkg + ".conversation")
    get_conv_template = conv.get_conv_template

    q = question
    if history is None and pixel_values is not None and "<image>" not in q:
        q = "<image>\n" + q

    img_context_token_id = tokenizer.convert_tokens_to_ids(img_ctx)
    model.img_context_token_id = img_context_token_id

    template = get_conv_template(model.template)
    template.system_message = model.system_message
    hist = [] if history is None else history
    for old_question, old_answer in hist:
        template.append_message(template.roles[0], old_question)
        template.append_message(template.roles[1], old_answer)
    template.append_message(template.roles[0], q)
    template.append_message(template.roles[1], None)
    query = template.get_prompt()

    for num_patches in num_patches_list:
        image_tokens = img_start + img_ctx * model.num_image_token * num_patches + img_end
        query = query.replace("<image>", image_tokens, 1)

    print("\n=== InternVL preflight (same as model.chat before generate) ===", flush=True)
    print(f"model.num_image_token (ViT tokens per patch before LLM): {model.num_image_token}", flush=True)
    print(
        "expanded query (collapsed IMG_CONTEXT runs):\n",
        collapse_img_context_runs(query),
        "\n",
        sep="",
        flush=True,
    )

    model_inputs = tokenizer(query, return_tensors="pt")
    input_ids = model_inputs["input_ids"]
    attn = model_inputs.get("attention_mask")
    flat_ids = input_ids[0].tolist()
    print(f"input_ids shape: {tuple(input_ids.shape)}", flush=True)
    if attn is not None:
        print(f"attention_mask shape: {tuple(attn.shape)}", flush=True)

    n_img_ctx = sum(1 for t in flat_ids if t == img_context_token_id)
    expected_ctx = model.num_image_token * sum(num_patches_list)
    print(
        f"<IMG_CONTEXT> token id={img_context_token_id}; count in input_ids={n_img_ctx} "
        f"(expected {expected_ctx} = num_image_token × sum(num_patches_list))",
        flush=True,
    )

    pieces = tokenizer.convert_ids_to_tokens(flat_ids)
    print("\n--- Full sequence: idx\\tid\\tpiece [IMG_CTX marked] ---", flush=True)
    for i, (tid, piece) in enumerate(zip(flat_ids, pieces)):
        mark = "  <-- IMG_CONTEXT" if tid == img_context_token_id else ""
        piece_esc = piece.replace("\n", "\\n") if isinstance(piece, str) else piece
        print(f"{i}\t{tid}\t{piece_esc}{mark}", flush=True)

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    pv = pixel_values.to(device=device, dtype=dtype)
    print("\n--- Vision tower + mlp1 (`extract_feature`), matches generate() injection source ---", flush=True)
    print(f"pixel_values shape (N_patch_rows, C, H, W): {tuple(pv.shape)}", flush=True)
    with torch.no_grad():
        vit = model.extract_feature(pv)
    print(f"vit_embeds shape [N_rows, n_tokens_per_row, llm_hidden]: {tuple(vit.shape)}", flush=True)
    print(f"vit_embeds numel flattened to rows: {vit.numel() // vit.shape[-1]} (should match {n_img_ctx})", flush=True)
    print("=== end preflight ===\n", flush=True)


def hf_id_piece_lines(tokenizer, text: str, *, add_special_tokens: bool = False) -> list[str]:
    """Lines matching foundation_token_io HF rows: id<TAB>subword_piece."""
    ids = tokenizer.encode(text, add_special_tokens=add_special_tokens)
    lines = []
    for tid in ids:
        subtokens = tokenizer.convert_ids_to_tokens([tid])
        piece = subtokens[0] if subtokens else ""
        piece_esc = piece.replace("\t", " ").replace("\r", "\\r").replace("\n", "\\n")
        lines.append(f"{tid}\t{piece_esc}")
    return lines


def build_foundation_token_io_document(
    tokenizer,
    *,
    question: str,
    num_patches_list: list[int],
    vision_slots_per_patch: int,
    response1: str,
    response2: str | None,
    follow_text: str | None,
) -> str:
    """Single transcript similar to `foundation_token_io.txt` (HF tokenizer + synthetic vision slots)."""
    lines_out: list[str] = []

    lines_out.append("User:")
    lines_out.append(question)
    lines_out.append("")
    lines_out.append("Assistant:")
    lines_out.append(response1)
    if response2 is not None:
        lines_out.append("")
        lines_out.append("User:")
        lines_out.append(follow_text or "")
        lines_out.append("")
        lines_out.append("Assistant:")
        lines_out.append(response2)
    lines_out.append("")

    lines_out.append("--- HF_TOKENIZER (full string passed to model.chat turn 1; id\\tsubword_piece) ---")
    lines_out.extend(hf_id_piece_lines(tokenizer, question))
    lines_out.append("")
    lines_out.append("")

    total_vis = sum(n * vision_slots_per_patch for n in num_patches_list)
    lines_out.append(
        "--- PREFILL_STYLE (per-frame text tokens + synthetic vision rows; id -1 = embedding/KV slot, not vocab) ---"
    )
    lines_out.append(
        f"# sum(num_patches_list)*{vision_slots_per_patch} = {total_vis} placeholders total "
        f"(VISION_KV_SLOT denominator is per-frame chunk, like mtmd per <image>, not this global sum)"
    )
    lines_out.append("--- PREFILL ---")

    for fi, n_patch in enumerate(num_patches_list):
        chunk = f"Frame{fi + 1}: <image>\n"
        lines_out.append(f"## FRAME {fi + 1} TEXT n_chars={len(chunk)}")
        lines_out.extend(hf_id_piece_lines(tokenizer, chunk))
        n_slots = n_patch * vision_slots_per_patch
        lines_out.append(f"## FRAME {fi + 1} IMAGE n_placeholder_tokens={n_slots}")
        for j in range(n_slots):
            lines_out.append(f"-1\t<VISION_KV_SLOT {j + 1}/{n_slots}>")

    tail_start = "".join(f"Frame{i + 1}: <image>\n" for i in range(len(num_patches_list)))
    tail_q = question[len(tail_start) :] if question.startswith(tail_start) else ""
    if tail_q:
        lines_out.append("## TAIL_QUESTION TEXT")
        lines_out.extend(hf_id_piece_lines(tokenizer, tail_q))
    lines_out.append("")
    lines_out.append("")

    lines_out.append("--- DECODE (assistant turn 1; id\\tpiece from tokenizer.encode on saved reply text) ---")
    lines_out.extend(hf_id_piece_lines(tokenizer, response1))
    if response2 is not None:
        lines_out.append("")
        lines_out.append("")
        lines_out.append("--- DECODE (assistant turn 2) ---")
        lines_out.extend(hf_id_piece_lines(tokenizer, response2))

    lines_out.append("")
    return "\n".join(lines_out)


def load_model(args: argparse.Namespace):
    path = args.model
    model = AutoModel.from_pretrained(
        path,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).eval()
    if torch.cuda.is_available():
        model = model.cuda()
    return model


def parse_args():
    p = argparse.ArgumentParser(description="InternVL3-1B video chat (HF Transformers)")
    p.add_argument("--video", type=Path, default=DEFAULT_VIDEO, help="Path to mp4")
    p.add_argument("--model", type=str, default="OpenGVLab/InternVL3-1B")
    p.add_argument("--num-segments", type=int, default=8, help="Uniform temporal samples (README uses 8 in video example)")
    p.add_argument("--max-num", type=int, default=1, help="max_num tiles per frame (README video uses 1)")
    p.add_argument("--input-size", type=int, default=448)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--do-sample", action="store_true")
    p.add_argument("--bf16", action="store_true", default=True, help="default bfloat16 on CUDA")
    p.add_argument("--no-bf16", action="store_false", dest="bf16")
    p.add_argument(
        "--question",
        type=str,
        default=None,
        help="Appended after Frame1: <image>... prefix; default describes surveillance clip",
    )
    p.add_argument(
        "--results-root",
        type=Path,
        default=RESULTS_PARENT,
        help="Directory under which a timestamped run folder is created",
    )
    p.add_argument(
        "--vision-slots-per-patch",
        type=int,
        default=256,
        help="Synthetic KV slots per visual patch (heuristic; InternVL IMG_CONTEXT-style ~256 per 448 tile)",
    )
    p.add_argument(
        "--dump-model-inputs",
        action="store_true",
        help="Print InternVL-expanded prompt, tokenizer(input_ids) after <image>→<IMG_CONTEXT>*, and extract_feature shape.",
    )
    p.add_argument(
        "--single-turn",
        action="store_true",
        help="Only first question/answer (omit second turn files and second DECODE block)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    _require_transformers_v4()
    if not args.video.is_file():
        sys.exit(f"missing video file: {args.video}")

    if not torch.cuda.is_available():
        print("CUDA not available — loading on CPU will likely OOM for InternVL3-1B.", file=sys.stderr)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, use_fast=False)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = args.results_root.resolve() / f"{args.video.stem}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"results dir: {run_dir}", flush=True)

    print("loading model...", flush=True)
    model = load_model(args)

    print(f"loading video: {args.video} (segments={args.num_segments}, max_num={args.max_num})", flush=True)
    pixel_values, num_patches_list = load_video(
        args.video,
        input_size=args.input_size,
        max_num=args.max_num,
        num_segments=args.num_segments,
    )
    pv_dtype = torch.bfloat16 if args.bf16 else torch.float16
    if torch.cuda.is_available():
        pixel_values = pixel_values.to(dtype=pv_dtype, device="cuda")

    generation_config = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature if args.do_sample else None,
    )
    # Drop None keys for cleanliness
    generation_config = {k: v for k, v in generation_config.items() if v is not None}

    video_prefix = "".join(f"Frame{i + 1}: <image>\n" for i in range(len(num_patches_list)))
    default_q = "Briefly describe what happens in this surveillance video."
    tail_q = args.question or default_q
    question = video_prefix + tail_q

    print("\n--- TOKENIZE(question): id\\tsubword_piece [add_special_tokens=False] ---", flush=True)
    _q_ids = tokenizer.encode(question, add_special_tokens=False)
    print(f"n_tokens={len(_q_ids)}", flush=True)
    for line in hf_id_piece_lines(tokenizer, question):
        print(line, flush=True)

    print("\n--- User (prompt prefix preview) ---", flush=True)
    print(video_prefix + "[... tail question ...]\n", flush=True)
    print("--- full question (complete) ---\n", question, "\n", sep="", flush=True)

    meta = {
        "utc_run_id": ts,
        "model": args.model,
        "video": str(args.video.resolve()),
        "num_segments": args.num_segments,
        "max_num": args.max_num,
        "input_size": args.input_size,
        "generation_config": generation_config,
        "num_patches_list": num_patches_list,
        "note": "Vision frames are not encoded as text tokens; pixel_values passed separately to model.chat.",
    }
    write_plain(run_dir / "meta.json", json.dumps(meta, indent=2, ensure_ascii=False))
    write_plain(run_dir / "01_user_turn1_question.txt", question)
    n_q1 = write_token_tsv(run_dir / "01_user_turn1_question.tokens.tsv", tokenizer, question)

    if args.dump_model_inputs:
        print_internvl_preflight_after_vision_processor(
            model,
            tokenizer,
            question=question,
            num_patches_list=num_patches_list,
            pixel_values=pixel_values,
            history=None,
        )

    response, history = model.chat(
        tokenizer,
        pixel_values,
        question,
        generation_config,
        num_patches_list=num_patches_list,
        history=None,
        return_history=True,
    )
    print("\n--- Assistant ---\n", response, sep="", flush=True)
    write_plain(run_dir / "02_assistant_turn1_response.txt", response)
    n_a1 = write_token_tsv(run_dir / "02_assistant_turn1_response.tokens.tsv", tokenizer, response)

    follow = "Summarize any notable events or objects in order of time."
    response2: str | None = None
    n_q2 = 0
    n_a2 = 0
    if not args.single_turn:
        write_plain(run_dir / "03_user_turn2_question.txt", follow)
        n_q2 = write_token_tsv(run_dir / "03_user_turn2_question.tokens.tsv", tokenizer, follow)

        response2, history = model.chat(
            tokenizer,
            pixel_values,
            follow,
            generation_config,
            num_patches_list=num_patches_list,
            history=history,
            return_history=True,
        )
        print("\n--- Assistant (2nd turn) ---\n", response2, sep="", flush=True)
        write_plain(run_dir / "04_assistant_turn2_response.txt", response2)
        n_a2 = write_token_tsv(run_dir / "04_assistant_turn2_response.tokens.tsv", tokenizer, response2)

    summary_lines = [
        "# token counts (HF tokenizer.encode on exact strings shown to console / saved .txt)",
        f"user_turn1_question_tokens={n_q1}",
        f"assistant_turn1_response_tokens={n_a1}",
        f"user_turn2_question_tokens={n_q2}",
        f"assistant_turn2_response_tokens={n_a2}",
    ]
    write_plain(run_dir / "00_token_counts.summary.txt", "\n".join(summary_lines) + "\n")

    fio_doc = build_foundation_token_io_document(
        tokenizer,
        question=question,
        num_patches_list=num_patches_list,
        vision_slots_per_patch=args.vision_slots_per_patch,
        response1=response,
        response2=response2,
        follow_text=follow if not args.single_turn else None,
    )
    write_plain(run_dir / "foundation_token_io.txt", fio_doc)
    print("\n========== foundation_token_io.txt (also saved under results dir) ==========\n", flush=True)
    print(fio_doc, flush=True)


if __name__ == "__main__":
    main()
