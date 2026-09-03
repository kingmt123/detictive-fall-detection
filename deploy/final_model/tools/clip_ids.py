"""读取可版本化、可审计的 clip ID 选择列表。"""
from collections import Counter
from collections.abc import Iterable
from pathlib import Path


def load_clip_ids(
    cli_clip_ids: Iterable[str] | None,
    clip_id_files: Iterable[Path] | None,
) -> set[str] | None:
    values = list(cli_clip_ids or ())
    for clip_id_file in clip_id_files or ():
        for raw_line in Path(clip_id_file).read_text(encoding="utf-8").splitlines():
            value = raw_line.strip()
            if value and not value.startswith("#"):
                values.append(value)
    if not values:
        return None
    if any(not value for value in values):
        raise ValueError("clip_id 必须是非空字符串")
    counts = Counter(values)
    duplicates = sorted(value for value, count in counts.items() if count > 1)
    if duplicates:
        raise ValueError(f"canary clip_id 重复: {duplicates}")
    return set(values)
