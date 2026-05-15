#!/usr/bin/env python3
from __future__ import annotations

import argparse
import difflib
import json
import re
from dataclasses import dataclass
from pathlib import Path

from transformers import AutoTokenizer


DEFAULT_HF_TOKENIZER = Path(
    "/root/.cache/huggingface/hub/models--OpenGVLab--InternVL3-1B/"
    "snapshots/4415a3b810e636d11dfa86b0e9ba40bb00535aa8"
)

INTERNVL_SYSTEM_MESSAGE = (
    "你是书生·万象，英文名是InternVL，是由上海人工智能实验室、清华大学及多家合作单位联合开发的多模态大语言模型。"
)


@dataclass(frozen=True)
class TokenEntry:
    token_id: int
    piece: str

    def line(self) -> str:
        return f"{self.token_id}\t{self.piece}"

    def key(self) -> str:
        return self.line()


def collapse_img_context(query: str) -> str:
    marker = "<IMG_CONTEXT>"
    pattern = re.compile(f"(?:{re.escape(marker)})+")

    def repl(match: re.Match[str]) -> str:
        return f"{marker}x{len(match.group(0)) // len(marker)}"

    return pattern.sub(repl, query)


def load_manifest(result_dir: Path) -> dict:
    manifest_path = result_dir / "media_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing manifest: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def hf_video_question(manifest: dict, *, frame_style: str) -> str:
    num_patches = manifest.get("num_patches_list") or [1] * int(manifest.get("num_segments", 0))
    raw_prompt = manifest.get("raw_prompt") or "Describe this video briefly."
    if frame_style == "internvl":
        prefix = "".join(f"Frame{i + 1}: <image>\n" for i in range(len(num_patches)))
    elif frame_style == "legacy_spaced":
        prefix = "".join(f"Frame {i + 1}: <image>\n" for i in range(len(num_patches)))
    else:
        raise ValueError(f"unknown frame style: {frame_style}")
    return prefix + raw_prompt


def hf_chat_query(question: str, num_patches_list: list[int], *, image_tokens_per_patch: int) -> str:
    query = (
        f"<|im_start|>system\n{INTERNVL_SYSTEM_MESSAGE}<|im_end|>\n"
        f"<|im_start|>user\n{question}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    for num_patches in num_patches_list:
        image_tokens = "<img>" + ("<IMG_CONTEXT>" * image_tokens_per_patch * int(num_patches)) + "</img>"
        query = query.replace("<image>", image_tokens, 1)
    return query


def tokenize(tokenizer, text: str) -> list[TokenEntry]:
    ids = tokenizer.encode(text, add_special_tokens=False)
    pieces = tokenizer.convert_ids_to_tokens(ids)
    return [
        TokenEntry(int(token_id), str(piece).replace("\t", " ").replace("\n", "\\n"))
        for token_id, piece in zip(ids, pieces)
    ]


def normalize_hf_entries(entries: list[TokenEntry], *, img_context_id: int) -> list[TokenEntry]:
    out: list[TokenEntry] = []
    slot_idx = 0
    for entry in entries:
        if entry.token_id == img_context_id:
            slot_idx += 1
            out.append(TokenEntry(-1, f"<VISION_KV_SLOT {slot_idx}>"))
        else:
            slot_idx = 0
            out.append(entry)
    return out


def parse_current_prefill(path: Path) -> list[TokenEntry]:
    if not path.exists():
        raise FileNotFoundError(f"missing current token trace: {path}")
    entries: list[TokenEntry] = []
    in_prefill = False
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if raw.startswith("--- PREFILL"):
            in_prefill = True
            continue
        if raw.startswith("--- DECODE"):
            break
        if not in_prefill or raw.startswith("#") or raw.startswith("##") or not raw.strip():
            continue
        match = re.match(r"^(-?\d+)\t(.+)$", raw)
        if match:
            entries.append(TokenEntry(int(match.group(1)), match.group(2)))
    return entries


def normalized_compare_lines(entries: list[TokenEntry]) -> list[str]:
    lines: list[str] = []
    slot_run_idx = 0
    for entry in entries:
        if entry.token_id == -1:
            slot_run_idx += 1
            lines.append("-1")
        else:
            slot_run_idx = 0
            lines.append(str(entry.token_id))
    return lines


def first_difference(left: list[str], right: list[str]) -> tuple[int, str | None, str | None] | None:
    for idx, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return idx, a, b
    if len(left) != len(right):
        idx = min(len(left), len(right))
        return idx, left[idx] if idx < len(left) else None, right[idx] if idx < len(right) else None
    return None


def strip_hf_system(entries: list[TokenEntry]) -> list[TokenEntry]:
    # HF InternVL conversation template always starts with:
    # <|im_start|>system\n...<|im_end|>\n
    for idx in range(len(entries) - 1):
        if entries[idx].piece == "<|im_end|>" and entries[idx + 1].piece in {"Ċ", "\\n"}:
            return entries[idx + 2 :]
    return entries


def write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare HF InternVL video prompt tokens with current mtmd trace.")
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--hf-tokenizer", type=Path, default=DEFAULT_HF_TOKENIZER)
    parser.add_argument("--image-tokens-per-patch", type=int, default=256)
    args = parser.parse_args()

    manifest = load_manifest(args.result_dir)
    num_patches_list = [int(v) for v in manifest.get("num_patches_list", [])]
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.hf_tokenizer),
        trust_remote_code=True,
        local_files_only=True,
        use_fast=False,
    )
    img_context_id = int(tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>"))

    question_internvl = hf_video_question(manifest, frame_style="internvl")
    query_internvl = hf_chat_query(
        question_internvl,
        num_patches_list,
        image_tokens_per_patch=args.image_tokens_per_patch,
    )
    hf_entries = tokenize(tokenizer, query_internvl)
    hf_norm = normalize_hf_entries(hf_entries, img_context_id=img_context_id)
    hf_norm_no_system = strip_hf_system(hf_norm)

    question_legacy_spaced = hf_video_question(manifest, frame_style="legacy_spaced")
    query_legacy_spaced = hf_chat_query(
        question_legacy_spaced,
        num_patches_list,
        image_tokens_per_patch=args.image_tokens_per_patch,
    )
    hf_legacy_spaced_norm = normalize_hf_entries(
        tokenize(tokenizer, query_legacy_spaced),
        img_context_id=img_context_id,
    )
    hf_legacy_spaced_norm_no_system = strip_hf_system(hf_legacy_spaced_norm)

    current_entries = parse_current_prefill(args.result_dir / "foundation_inference_tokens.txt")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "expected_hf_internvl_query_collapsed.txt").write_text(
        collapse_img_context(query_internvl) + "\n",
        encoding="utf-8",
    )
    (args.out_dir / "expected_hf_legacy_spaced_frame_query_collapsed.txt").write_text(
        collapse_img_context(query_legacy_spaced) + "\n",
        encoding="utf-8",
    )
    write_lines(args.out_dir / "expected_hf_internvl_tokens.tsv", [entry.line() for entry in hf_entries])
    write_lines(args.out_dir / "expected_hf_internvl_normalized.tsv", [entry.line() for entry in hf_norm])
    write_lines(
        args.out_dir / "expected_hf_internvl_normalized_no_system.tsv",
        [entry.line() for entry in hf_norm_no_system],
    )
    write_lines(
        args.out_dir / "expected_hf_legacy_spaced_frame_normalized_no_system.tsv",
        [entry.line() for entry in hf_legacy_spaced_norm_no_system],
    )
    write_lines(args.out_dir / "current_llamacpp_prefill_tokens.tsv", [entry.line() for entry in current_entries])

    current_cmp = normalized_compare_lines(current_entries)
    hf_cmp = normalized_compare_lines(hf_norm)
    hf_no_system_cmp = normalized_compare_lines(hf_norm_no_system)
    hf_legacy_spaced_no_system_cmp = normalized_compare_lines(hf_legacy_spaced_norm_no_system)

    diff_global = list(difflib.unified_diff(hf_cmp, current_cmp, fromfile="hf_internvl", tofile="current", n=8))
    diff_no_system = list(
        difflib.unified_diff(hf_no_system_cmp, current_cmp, fromfile="hf_internvl_no_system", tofile="current", n=8)
    )
    diff_legacy_spaced_no_system = list(
        difflib.unified_diff(
            hf_legacy_spaced_no_system_cmp,
            current_cmp,
            fromfile="hf_legacy_spaced_frame_no_system",
            tofile="current",
            n=8,
        )
    )
    write_lines(args.out_dir / "diff_hf_internvl_vs_current.diff", diff_global)
    write_lines(args.out_dir / "diff_hf_internvl_no_system_vs_current.diff", diff_no_system)
    write_lines(
        args.out_dir / "diff_hf_legacy_spaced_frame_no_system_vs_current.diff",
        diff_legacy_spaced_no_system,
    )

    report = [
        "# Video Token Comparison",
        "",
        f"- result_dir: `{args.result_dir}`",
        f"- hf_tokenizer: `{args.hf_tokenizer}`",
        f"- num_patches_list: `{num_patches_list}`",
        f"- image_tokens_per_patch: `{args.image_tokens_per_patch}`",
        f"- IMG_CONTEXT token id: `{img_context_id}`",
        "",
        "## Counts",
        "",
        f"- HF InternVL full normalized entries: `{len(hf_norm)}`",
        f"- HF InternVL normalized entries after stripping system prompt: `{len(hf_norm_no_system)}`",
        f"- HF legacy-spaced-frame normalized entries after stripping system prompt: `{len(hf_legacy_spaced_norm_no_system)}`",
        f"- current llama.cpp/mtmd prefill entries: `{len(current_entries)}`",
        f"- current vision slots: `{sum(1 for entry in current_entries if entry.token_id == -1)}`",
        f"- HF IMG_CONTEXT slots: `{sum(1 for entry in hf_entries if entry.token_id == img_context_id)}`",
        "",
        "## First Differences",
        "",
    ]
    checks = [
        ("HF InternVL full vs current", first_difference(hf_cmp, current_cmp)),
        ("HF InternVL without system vs current", first_difference(hf_no_system_cmp, current_cmp)),
        (
            "HF legacy spaced frame style without system vs current",
            first_difference(hf_legacy_spaced_no_system_cmp, current_cmp),
        ),
    ]
    for name, diff in checks:
        report.append(f"### {name}")
        if diff is None:
            report.append("- no token-level difference after normalization")
        else:
            idx, expected, actual = diff
            report.append(f"- first different index: `{idx}`")
            report.append(f"- expected: `{expected}`")
            report.append(f"- actual: `{actual}`")
        report.append("")

    report.extend(
        [
            "## Interpretation",
            "",
            "1. HF InternVL `model.chat()` prepends the InternVL system message before the user turn.",
            "2. HF video examples use `Frame1: <image>` without a space after `Frame`; older runner logs may use `Frame 1: <__media__>`.",
            "3. HF uses `<IMG_CONTEXT>` token ids as placeholders before replacing those positions with vision embeddings; current mtmd emits direct image chunks represented here as `-1` slots. These are comparable only after normalizing both to `VISION_KV_SLOT`.",
            "4. Current runner builds should match the HF InternVL frame label after stripping only the system prompt.",
            "",
            "## Files",
            "",
            "- `expected_hf_internvl_query_collapsed.txt`",
            "- `expected_hf_internvl_normalized.tsv`",
            "- `expected_hf_internvl_normalized_no_system.tsv`",
            "- `expected_hf_legacy_spaced_frame_normalized_no_system.tsv`",
            "- `current_llamacpp_prefill_tokens.tsv`",
            "- `diff_hf_internvl_vs_current.diff`",
            "- `diff_hf_internvl_no_system_vs_current.diff`",
            "- `diff_hf_legacy_spaced_frame_no_system_vs_current.diff`",
        ]
    )
    (args.out_dir / "README.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
