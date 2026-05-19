from __future__ import annotations

from dataclasses import dataclass


MEDIA_MARKER = "<__media__>"
DEFAULT_PROMPT_FORMAT = "internvl3"


def normalize_prompt_format(prompt_format: str | None) -> str:
    if prompt_format is None or not prompt_format.strip():
        return DEFAULT_PROMPT_FORMAT
    normalized = prompt_format.strip().lower().replace("-", "_").replace(".", "_")
    aliases = {
        "internvl": "internvl3",
        "internvl3_instruct": "internvl3",
        "qwen25vl": "qwen2_5_vl",
        "qwen2_5vl": "qwen2_5_vl",
        "qwen2_5_vl": "qwen2_5_vl",
        "qwen2_5_vl_instruct": "qwen2_5_vl",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"internvl3", "qwen2_5_vl"}:
        raise ValueError(f"unsupported prompt format: {prompt_format!r}")
    return normalized


@dataclass(frozen=True)
class PromptFormatter:
    name: str
    media_marker: str = MEDIA_MARKER
    video_frame_prefix: str = "Frame"
    video_frame_separator: str = ": "
    multi_image_prefix: str = "Image"
    multi_image_index_separator: str = "-"
    multi_image_separator: str = ": "

    def image_prompt(self, raw_prompt: str, *, n_images: int = 1) -> str:
        return (self.media_marker * max(int(n_images), 1)) + "\n" + raw_prompt

    def multi_image_prompt(self, num_images: int, raw_prompt: str) -> str:
        parts: list[str] = []
        for image_i in range(max(int(num_images), 0)):
            parts.append(
                f"{self.multi_image_prefix}{self.multi_image_index_separator}{image_i + 1}"
                f"{self.multi_image_separator}{self.media_marker}\n"
            )
        return "".join(parts) + raw_prompt

    def video_prompt(self, num_patches_list: list[int], raw_prompt: str) -> str:
        parts: list[str] = []
        for frame_i, n_patches in enumerate(num_patches_list):
            parts.append(
                f"{self.video_frame_prefix}{frame_i + 1}{self.video_frame_separator}"
                + (self.media_marker * max(int(n_patches), 1))
                + "\n"
            )
        return "".join(parts) + raw_prompt


_FORMATTERS = {
    "internvl3": PromptFormatter(name="internvl3"),
    # Qwen2.5-VL support currently uses the same abstract mtmd media marker at
    # the runner boundary. Model-specific special-token expansion remains in
    # llama.cpp/mtmd, while this registry isolates naming and frame-layout policy.
    "qwen2_5_vl": PromptFormatter(name="qwen2_5_vl"),
}


def get_prompt_formatter(prompt_format: str | None = None) -> PromptFormatter:
    return _FORMATTERS[normalize_prompt_format(prompt_format)]
