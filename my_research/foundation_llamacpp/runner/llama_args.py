from __future__ import annotations

import shlex


def rope_shell_suffix(args) -> str:
    parts: list[str] = []
    rs = getattr(args, "rope_scaling", None)
    if rs:
        parts.append(f"--rope-scaling {shlex.quote(rs)}")
    if getattr(args, "rope_scale", None) is not None:
        parts.append(f"--rope-scale {shlex.quote(str(args.rope_scale))}")
    if getattr(args, "rope_freq_base", None) is not None:
        parts.append(f"--rope-freq-base {shlex.quote(str(args.rope_freq_base))}")
    if getattr(args, "rope_freq_scale", None) is not None:
        parts.append(f"--rope-freq-scale {shlex.quote(str(args.rope_freq_scale))}")
    if getattr(args, "yarn_orig_ctx", None) is not None:
        parts.append(f"--yarn-orig-ctx {shlex.quote(str(args.yarn_orig_ctx))}")
    if getattr(args, "yarn_ext_factor", None) is not None:
        parts.append(f"--yarn-ext-factor {shlex.quote(str(args.yarn_ext_factor))}")
    if getattr(args, "yarn_attn_factor", None) is not None:
        parts.append(f"--yarn-attn-factor {shlex.quote(str(args.yarn_attn_factor))}")
    if getattr(args, "yarn_beta_slow", None) is not None:
        parts.append(f"--yarn-beta-slow {shlex.quote(str(args.yarn_beta_slow))}")
    if getattr(args, "yarn_beta_fast", None) is not None:
        parts.append(f"--yarn-beta-fast {shlex.quote(str(args.yarn_beta_fast))}")
    return (" " + " ".join(parts)) if parts else ""

