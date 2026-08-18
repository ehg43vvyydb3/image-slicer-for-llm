#!/usr/bin/env python3
"""LLM에 올리기 좋게 이미지를 잘라주는 CLI.

전체 페이지 스크린샷처럼 세로로 긴 이미지를 그대로 올리면 LLM이 긴 변을 기준으로
축소해 버려서 글자를 읽을 수 없게 된다. 이 도구는 각 조각이 축소 한계 안에 들어오도록
잘라서 `<이미지이름>_output/` 폴더에 순서대로 저장한다.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from PIL import Image, ImageOps

# 세로로 아주 긴 전체 페이지 스크린샷도 열 수 있도록 한도를 올린다.
Image.MAX_IMAGE_PIXELS = 500_000_000

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff"}

EXTENSIONS = {"png": ".png", "jpeg": ".jpg", "webp": ".webp"}

# 출력 폴더는 소스 위치와 무관하게 항상 다운로드 폴더 밑에 만든다.
DOWNLOADS_DIR = Path.home() / "Downloads"


def default_outdir(stem: str) -> Path:
    return DOWNLOADS_DIR / f"{stem}_output"


def claude_tokens(width: int, height: int) -> int:
    """Claude 비전 토큰 근사치: 가로 x 세로 / 750."""
    return round(width * height / 750)


def openai_tokens(width: int, height: int) -> int:
    """512x512 타일 기반: base 85 + 타일당 170 (GPT-4o 계열 경험치)."""
    return 85 + 170 * math.ceil(width / 512) * math.ceil(height / 512)


def gemini_tokens(width: int, height: int) -> int:
    """768x768 타일 기반: 타일당 258 (비공식)."""
    return 258 * math.ceil(width / 768) * math.ceil(height / 768)


@dataclass(frozen=True)
class Preset:
    name: str
    max_edge: int  # 조각의 긴 변 상한
    max_pixels: int  # 조각의 총 픽셀 상한
    ideal_width: int | None  # 권장 조각 가로 폭 (None 이면 원본 폭 유지)
    tokens: Callable[[int, int], int]
    note: str


PRESETS = {
    # 긴 변 1568px / 약 1.15MP. 어떤 Claude 모델에 올려도 추가 축소가 일어나지 않는다.
    "claude": Preset(
        "claude", 1568, 1_150_000, None, claude_tokens,
        "모든 Claude 모델에서 축소 없이 인식",
    ),
    # 고해상도 입력을 지원하는 모델(Opus 4.7 이후, Sonnet 5 등) 기준.
    "claude-hires": Preset(
        "claude-hires", 2576, 3_750_000, None, claude_tokens,
        "고해상도 지원 Claude 모델 전용 (조각 수가 줄어듦)",
    ),
    # detail=high 파이프라인: 2048 박스에 맞춘 뒤 '짧은 변을 768px로' 리사이즈하고
    # 512x512 타일로 나눈다. 짧은 변이 768 이어야 재샘플링이 일어나지 않는다.
    "openai": Preset(
        "openai", 2048, 768 * 2048, 768, openai_tokens,
        "GPT-4o/GPT-5 계열. 짧은 변 768px 고정이 핵심",
    ),
    # 768x768 타일 기반. 공식 문서로 확인되지 않아 보수적으로 잡았다.
    "gemini": Preset(
        "gemini", 2304, 768 * 2304, 768, gemini_tokens,
        "Gemini 계열. 공식 미확인 수치라 보수적으로 설정",
    ),
}


@dataclass
class Tile:
    index: int
    row: int
    col: int
    box: tuple[int, int, int, int]  # (left, top, right, bottom)

    @property
    def width(self) -> int:
        return self.box[2] - self.box[0]

    @property
    def height(self) -> int:
        return self.box[3] - self.box[1]


# ---------------------------------------------------------------- 자르는 위치 계산


def row_variance(image: Image.Image) -> list[float]:
    """가로줄마다 픽셀 분산을 구한다. 값이 0에 가까우면 여백(빈 줄)이다."""
    gray = image.convert("L")
    height = gray.height

    # 폭 1로 줄이면 BOX 필터가 각 줄의 평균을 내준다.
    means = list(gray.resize((1, height), Image.BOX).getdata())
    squares = gray.point(lambda v: (v * v) // 255)
    square_means = list(squares.resize((1, height), Image.BOX).getdata())

    return [max(0.0, m2 * 255 - m * m) for m, m2 in zip(means, square_means)]


def split_points(total: int, tile: int, step: int) -> list[int]:
    """total 길이를 나누는 경계 좌표.

    한 조각에 다 들어가면 나누지 않는다. 나눠야 하면 구간 길이를 step 이하로 균등하게
    맞춘다. 여기에 겹침이 더해져도 조각 크기는 tile 을 넘지 않는다.
    """
    if total <= tile:
        return [0, total]
    count = math.ceil(total / step)
    segment = math.ceil(total / count)
    points = [min(total, i * segment) for i in range(count + 1)]
    points[0] = 0
    points[-1] = total
    return points


def best_blank_row(
    variance: list[float], target: int, lowest: int, threshold: float = 25.0
) -> int:
    """[lowest, target] 구간에서 가장 여백에 가까운 가로줄을 찾는다.

    글자 줄 한가운데를 자르지 않기 위한 보정이다. target 아래로는 내려가지 않으므로
    보정 후에도 조각이 계산된 최대 높이를 넘지 않는다. 마땅한 여백이 없으면 target 을
    그대로 돌려준다.
    """
    highest = min(target, len(variance) - 1)
    lowest = max(0, min(lowest, highest))

    best, best_variance = target, None
    # 원래 위치에서 위로 훑어서, 동점이면 원래 위치에 가까운 줄을 고른다.
    for row in range(highest, lowest - 1, -1):
        value = variance[row]
        if best_variance is None or value < best_variance:
            best, best_variance = row, value

    if best_variance is None or best_variance > threshold:
        return target
    return best


def snap_to_blank_rows(
    points: list[int],
    variance: list[float],
    window: int,
    threshold: float = 25.0,
) -> list[int]:
    """각 경계를 위쪽 window 픽셀 안의 여백 줄로 당긴다."""
    snapped = [points[0]]
    for point in points[1:-1]:
        low = max(snapped[-1] + 1, point - window)
        snapped.append(best_blank_row(variance, point, low, threshold))
    snapped.append(points[-1])
    return snapped


def plan_tiles(
    width: int,
    height: int,
    tile_width: int,
    tile_height: int,
    overlap: int,
    variance: list[float] | None,
    smart_window: int,
) -> list[Tile]:
    """이미지 전체를 겹침을 포함한 조각 목록으로 나눈다 (왼쪽→오른쪽, 위→아래)."""
    v_step = max(1, tile_height - overlap)
    h_step = max(1, tile_width - overlap)

    rows = split_points(height, tile_height, v_step)
    cols = split_points(width, tile_width, h_step)

    if variance is not None and len(rows) > 2:
        rows = snap_to_blank_rows(rows, variance, smart_window)

    tiles: list[Tile] = []
    index = 1
    for r in range(len(rows) - 1):
        top = rows[r] if r == 0 else max(0, rows[r] - overlap)
        bottom = rows[r + 1]
        for c in range(len(cols) - 1):
            left = cols[c] if c == 0 else max(0, cols[c] - overlap)
            right = cols[c + 1]
            tiles.append(Tile(index, r + 1, c + 1, (left, top, right, bottom)))
            index += 1
    return tiles


def compute_tile_size(
    width: int, height: int, preset_max_edge: int, max_pixels: int
) -> tuple[int, int]:
    """긴 변 제한과 총 픽셀 제한을 동시에 만족하는 조각 크기."""
    tile_width = min(width, preset_max_edge)
    tile_height = min(height, preset_max_edge, max(1, max_pixels // tile_width))
    return tile_width, tile_height


# ---------------------------------------------------------------- 입출력


def load_image(path: Path) -> Image.Image:
    image = Image.open(path)
    image = ImageOps.exif_transpose(image)  # 회전 정보가 있으면 실제 픽셀에 반영
    return image


def prepare_for_save(image: Image.Image, fmt: str) -> Image.Image:
    if fmt == "jpeg" and image.mode in ("RGBA", "LA", "P"):
        rgba = image.convert("RGBA")
        canvas = Image.new("RGB", rgba.size, (255, 255, 255))
        canvas.paste(rgba, mask=rgba.split()[-1])
        return canvas
    if fmt != "jpeg" and image.mode == "P":
        return image.convert("RGBA")
    return image


def save_params(fmt: str, quality: int) -> dict:
    if fmt == "png":
        return {"optimize": True}
    if fmt == "jpeg":
        return {"quality": quality, "optimize": True, "progressive": True}
    return {"quality": quality, "method": 6}


def human_size(num_bytes: int) -> str:
    for unit in ("B", "KB", "MB"):
        if num_bytes < 1024 or unit == "MB":
            return f"{num_bytes:.0f}{unit}" if unit == "B" else f"{num_bytes:.1f}{unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f}MB"


def pick_image_interactively() -> Path | None:
    """macOS Finder 파일 선택창을 띄운다. 경로를 손으로 타이핑/붙여넣기 하지 않아도 되니
    한글·공백이 섞인 파일명에서 셸 이스케이프 실수가 날 여지가 없다."""
    types = ", ".join(f'"{ext.lstrip(".")}"' for ext in sorted(IMAGE_SUFFIXES))
    script = f'POSIX path of (choose file with prompt "자를 이미지를 고르세요" of type {{{types}}})'
    try:
        result = subprocess.run(["osascript", "-e", script], capture_output=True, text=True)
    except FileNotFoundError:
        print("파일 선택창을 열 수 없습니다 (macOS 전용 기능입니다).", file=sys.stderr)
        return None
    if result.returncode != 0:
        return None  # 사용자가 취소함
    return Path(result.stdout.strip())


def clear_previous_output(outdir: Path, stem: str) -> int:
    removed = 0
    for path in outdir.iterdir():
        if not path.is_file():
            continue
        is_slice = path.stem.startswith(f"{stem}_") and path.suffix.lower() in IMAGE_SUFFIXES
        if is_slice or path.name == "manifest.json":
            path.unlink()
            removed += 1
    return removed


# ---------------------------------------------------------------- 실행


def slice_image(path: Path, args: argparse.Namespace) -> int:
    preset = PRESETS[args.preset]
    max_edge = args.max_edge or preset.max_edge
    max_pixels = int(args.max_pixels) if args.max_pixels else preset.max_pixels

    try:
        image = load_image(path)
    except Exception as exc:  # 손상된 파일, 지원하지 않는 포맷 등
        print(f"[{path.name}] 열 수 없습니다: {exc}", file=sys.stderr)
        return 1

    original_size = (image.width, image.height)
    scale = 1.0

    # 가로가 목표 폭을 넘는 경우: 기본은 폭을 줄여 맞추고, --wide split이면 세로로도 나눈다.
    # 목표 폭은 --width > 프리셋 권장값 > 긴 변 한계 순으로 정해진다.
    fit_width = min(args.width or preset.ideal_width or max_edge, max_edge)
    if args.wide == "fit" and image.width > fit_width:
        scale = fit_width / image.width
        new_size = (fit_width, max(1, round(image.height * scale)))
        image = image.resize(new_size, Image.LANCZOS)

    tile_width, tile_height = compute_tile_size(image.width, image.height, max_edge, max_pixels)

    overlap = args.overlap
    smallest_tile = min(tile_width, tile_height)
    if overlap >= smallest_tile:
        overlap = max(0, smallest_tile // 4)
        print(f"[{path.name}] 겹침이 조각 크기보다 커서 {overlap}px로 줄였습니다.")

    variance = row_variance(image) if args.smart_cut else None
    tiles = plan_tiles(
        image.width, image.height, tile_width, tile_height, overlap, variance, args.smart_window
    )

    columns = max(t.col for t in tiles)
    print(f"\n■ {path.name}  {original_size[0]}x{original_size[1]}")
    if scale != 1.0:
        print(f"  폭 축소: x{scale:.3f} → {image.width}x{image.height}")
    print(
        f"  프리셋 {preset.name} (긴 변 {max_edge}px / 최대 {max_pixels/1_000_000:.2f}MP)"
        f" · 조각 {tile_width}x{tile_height} · 겹침 {overlap}px"
        f" · 여백 맞춤 {'켬' if args.smart_cut else '끔'}"
    )
    print(f"  조각 {len(tiles)}개 ({len(tiles)//columns}행 x {columns}열)"
          f" · 조각당 최대 약 {max(preset.tokens(t.width, t.height) for t in tiles):,} 토큰")

    if args.dry_run:
        for tile in tiles:
            left, top, right, bottom = tile.box
            print(f"    {tile.index:02d}. y {top:>6}–{bottom:<6} x {left:>5}–{right:<5}"
                  f"  ({tile.width}x{tile.height})")
        return 0

    outdir = args.outdir or default_outdir(path.stem)
    if not prepare_outdir(outdir, path.stem, args.force):
        return 1

    entries = (
        (image.crop(tile.box), {"row": tile.row, "col": tile.col, "box": list(tile.box)})
        for tile in tiles
    )
    records, total_bytes = emit_slices(outdir, path.stem, entries, args, preset.tokens)

    if args.manifest:
        write_manifest(
            outdir,
            {
                "source": {
                    "path": str(path.resolve()),
                    "size": list(original_size),
                    "scale_applied": round(scale, 6),
                    "working_size": [image.width, image.height],
                },
                "preset": {"name": preset.name, "max_edge": max_edge, "max_pixels": max_pixels},
                "tile": {"width": tile_width, "height": tile_height, "overlap": overlap},
                "grid": {"rows": len(tiles) // columns, "columns": columns},
                "reading_order": "index 순서대로 위에서 아래로 (여러 열이면 왼쪽에서 오른쪽으로)",
                "slices": records,
            },
        )

    finish(outdir, total_bytes, args)
    return 0


def slice_url(url: str, args: argparse.Namespace) -> int:
    import web_capture  # playwright 는 URL 을 쓸 때만 필요하다

    preset = PRESETS[args.preset]
    max_edge = args.max_edge or preset.max_edge
    max_pixels = int(args.max_pixels) if args.max_pixels else preset.max_pixels

    width = min(args.width or preset.ideal_width or 1200, max_edge)
    tile_height = min(max_edge, max(1, max_pixels // width))

    overlap = args.overlap
    if overlap >= tile_height:
        overlap = max(0, tile_height // 4)

    url, unescaped = normalize_url(url)
    problem = url_problem(url)
    if problem:
        print(f"\n■ {url}\n  {problem}", file=sys.stderr)
        return 2

    print(f"\n■ {url}")
    if unescaped:
        print("  (주소에 섞여 있던 백슬래시를 제거했습니다)")
    print(
        f"  프리셋 {preset.name} (긴 변 {max_edge}px / 최대 {max_pixels/1_000_000:.2f}MP)"
        f" · 조각 {width}x{tile_height} · 겹침 {overlap}px"
        f" · 여백 맞춤 {'켬' if args.smart_cut else '끔'}"
    )

    try:
        result = web_capture.capture_slices(
            url,
            width=width,
            tile_height=tile_height,
            overlap=overlap,
            scale=args.scale,
            smart_window=args.smart_window,
            smart_cut=args.smart_cut,
            engine=args.engine,
            headless=not (args.show or args.pause),
            extra_wait=args.wait,
            timeout=args.timeout,
            click_texts=args.click,
            pause=args.pause,
        )
    except web_capture.CaptureError as exc:
        print(f"  캡처 실패: {exc}", file=sys.stderr)
        return 1

    if not result.slices:
        print("  캡처된 내용이 없습니다.", file=sys.stderr)
        return 1

    stem = web_capture.slugify(result.title, fallback="capture")
    print(f"  제목: {result.title or '(없음)'}")
    if len(result.slices) == 1:
        print("  참고: 페이지가 한 화면에 다 들어갔습니다. "
              "주소가 맞는지, 로그인이 필요한 페이지는 아닌지 확인해 보세요.")
    print(f"  조각 {len(result.slices)}개 · 조각당 최대 약 "
          f"{max(preset.tokens(s.image.width, s.image.height) for s in result.slices):,} 토큰")

    outdir = args.outdir or default_outdir(stem)
    if not prepare_outdir(outdir, stem, args.force):
        return 1

    entries = (
        (item.image, {"row": i, "col": 1, "page_y": [item.page_top, item.page_bottom]})
        for i, item in enumerate(result.slices, start=1)
    )
    records, total_bytes = emit_slices(outdir, stem, entries, args, preset.tokens)

    if args.manifest:
        write_manifest(
            outdir,
            {
                "source": {
                    "url": result.url,
                    "final_url": result.final_url,
                    "title": result.title,
                    "page_height": result.page_height,
                    "viewport": list(result.viewport),
                    "device_scale_factor": result.device_scale_factor,
                    "engine": args.engine,
                    "click": args.click or [],
                },
                "preset": {"name": preset.name, "max_edge": max_edge, "max_pixels": max_pixels},
                "tile": {"width": width, "height": tile_height, "overlap": overlap},
                "grid": {"rows": len(result.slices), "columns": 1},
                "reading_order": "index 순서대로 페이지 위에서 아래로",
                "slices": records,
            },
        )

    finish(outdir, total_bytes, args)
    return 0


def prepare_outdir(outdir: Path, stem: str, force: bool) -> bool:
    """출력 폴더를 준비한다. 계속 진행해도 되면 True."""
    if outdir.exists() and any(outdir.iterdir()):
        if not force:
            print(
                f"  출력 폴더가 이미 비어있지 않습니다: {outdir}\n"
                f"  기존 조각을 지우고 다시 만들려면 --force 를 붙이세요.",
                file=sys.stderr,
            )
            return False
        print(f"  기존 파일 {clear_previous_output(outdir, stem)}개 삭제")
    outdir.mkdir(parents=True, exist_ok=True)
    return True


def emit_slices(outdir: Path, stem: str, entries, args: argparse.Namespace, tokens):
    """조각 이미지를 저장하고 manifest 레코드와 총 용량을 돌려준다."""
    ext = EXTENSIONS[args.format]
    params = save_params(args.format, args.quality)
    records: list[dict] = []
    total_bytes = 0

    for index, (image, extra) in enumerate(entries, start=1):
        name = f"{stem}_{index:02d}{ext}"
        target = outdir / name
        prepare_for_save(image, args.format).save(target, format=args.format.upper(), **params)

        size_bytes = target.stat().st_size
        total_bytes += size_bytes
        records.append(
            {
                "file": name,
                "index": index,
                **extra,
                "size": [image.width, image.height],
                "bytes": size_bytes,
                "est_tokens": tokens(image.width, image.height),
            }
        )
        if size_bytes > 5 * 1024 * 1024:
            print(f"  주의: {name} 이 {human_size(size_bytes)} 입니다 "
                  f"(업로드 용량 제한에 걸릴 수 있음, --format jpeg 권장)")

    return records, total_bytes


def write_manifest(outdir: Path, payload: dict) -> None:
    (outdir / "manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def finish(outdir: Path, total_bytes: int, args: argparse.Namespace) -> None:
    print(f"  저장 완료 → {outdir}  (총 {human_size(total_bytes)})")
    if args.open:
        subprocess.run(["open", str(outdir)], check=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="imgslice",
        description="세로로 긴 스크린샷을 LLM이 축소 없이 읽을 수 있는 크기로 잘라줍니다.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "예시:\n"
            "  imgslice --file                   파일 선택창에서 이미지를 골라서 자르기\n"
            "  imgslice page.png                 page_output/ 에 조각 저장\n"
            "  imgslice *.png --force            여러 장 한 번에, 기존 결과 덮어쓰기\n"
            "  imgslice page.png -p claude-hires 고해상도 모델용 (조각 수 감소)\n"
            "  imgslice page.png --dry-run       자르는 위치만 미리 확인\n"
            "\n"
            "  imgslice 'https://example.com/글'  URL을 직접 캡처해서 자르기\n"
            "  imgslice 'https://a.com/p' --click 더보기   캡처 전에 '더보기' 버튼을 모두 클릭\n"
            "  imgslice 'https://a.com/p' --pause          창을 띄워 직접 클릭/로그인 후 엔터로 캡처\n"
            "                                              (자동 클릭이 봇 탐지에 막히는 사이트용)\n"
            "\n"
            "  주소는 작은따옴표로 감싸세요.\n"
            "    O   imgslice 'https://a.com/p?x=1&y=2'\n"
            "    O   imgslice 'https://a.com/p\\?x=1\\&y=2'   (백슬래시는 알아서 제거함)\n"
            "    X   imgslice https://a.com/p?x=1&y=2       (셸이 & 에서 잘라먹음)\n"
            "\n"
            "  스크롤하며 화면 단위로 찍으므로 길이 제한도, lazy loading 문제도 없습니다.\n"
            "  모바일(m.) 페이지는 --width 500 처럼 좁게 주면 레이아웃이 자연스럽습니다.\n"
        ),
    )
    parser.add_argument(
        "images", nargs="*", metavar="이미지|URL",
        help="자를 이미지 경로 또는 http(s) 주소",
    )
    parser.add_argument(
        "--file", action="store_true",
        help="파일 선택창(Finder)에서 이미지를 골라 자르기 (경로/URL 인자 대신 사용)",
    )
    parser.add_argument(
        "-p", "--preset", choices=sorted(PRESETS), default="claude",
        help="대상 서비스 프리셋 (기본: claude). --list-presets 로 상세 확인",
    )
    parser.add_argument("--list-presets", action="store_true", help="프리셋 목록과 수치를 보여줍니다")
    parser.add_argument(
        "--width", type=int,
        help="조각 가로 폭. URL은 뷰포트 폭, 로컬 파일은 이 폭으로 축소 (기본: 프리셋 권장값)",
    )
    parser.add_argument("--max-edge", type=int, help="조각의 긴 변 최대 픽셀 (프리셋 값 덮어쓰기)")
    parser.add_argument("--max-pixels", type=float, help="조각의 총 픽셀 상한 (예: 1150000)")
    parser.add_argument("-o", "--overlap", type=int, default=60, help="조각 간 겹침 픽셀 (기본: 60)")
    parser.add_argument(
        "--wide", choices=["fit", "split"], default="fit",
        help="가로가 한계보다 넓을 때: fit=폭 축소(기본), split=열로 분할",
    )
    parser.add_argument(
        "--no-smart-cut", dest="smart_cut", action="store_false",
        help="글자 줄을 피해 여백에서 자르는 보정을 끕니다",
    )
    parser.add_argument("--smart-window", type=int, default=80, help="여백을 찾는 탐색 범위 (기본: 80px)")
    parser.add_argument("-f", "--format", choices=list(EXTENSIONS), default="png", help="출력 포맷")
    parser.add_argument("-q", "--quality", type=int, default=92, help="jpeg/webp 품질 (기본: 92)")
    parser.add_argument("--outdir", type=Path, help="출력 폴더 직접 지정")
    parser.add_argument("--force", action="store_true", help="기존 출력 파일을 지우고 다시 생성")
    parser.add_argument(
        "--no-manifest", dest="manifest", action="store_false", help="manifest.json 을 만들지 않음"
    )
    parser.add_argument("--dry-run", action="store_true", help="파일을 쓰지 않고 계획만 출력 (URL 모드에서는 무시)")
    parser.add_argument("--open", action="store_true", help="완료 후 Finder로 출력 폴더 열기")

    web = parser.add_argument_group("URL 캡처 옵션")
    web.add_argument("--scale", type=int, default=2, help="캡처 배율. 2면 2배로 찍어 줄여서 더 선명 (기본: 2)")
    web.add_argument(
        "--engine", choices=["chromium", "firefox", "webkit"], default="chromium",
        help="캡처에 쓸 브라우저 엔진 (기본: chromium)",
    )
    web.add_argument("--show", action="store_true", help="브라우저 창을 띄워서 진행 과정 보기")
    web.add_argument(
        "--click", action="append", metavar="텍스트",
        help="캡처 전 지정한 텍스트를 가진 요소를 모두 클릭 (예: --click 더보기). 여러 번 지정 가능",
    )
    web.add_argument(
        "--pause", action="store_true",
        help="브라우저 창을 띄우고 캡처 전에 멈춰서, 더보기 클릭·로그인 등을 직접 할 시간을 줌 (엔터로 계속)",
    )
    web.add_argument("--wait", type=float, default=0.0, help="캡처 전 추가 대기 초 (기본: 0)")
    web.add_argument("--timeout", type=float, default=60.0, help="페이지 로딩 제한 초 (기본: 60)")
    return parser


def is_url(value: str) -> bool:
    """`스킴://` 형태면 주소로 본다. http(s) 여부는 url_problem 에서 걸러낸다."""
    scheme, separator, _ = value.partition("://")
    return bool(separator) and scheme.isascii() and scheme.isalnum()


def normalize_url(url: str) -> tuple[str, bool]:
    """셸 이스케이프로 섞여 들어온 백슬래시를 걷어낸다.

    주소에 백슬래시가 정상적으로 들어가는 경우는 없다(RFC 3986). 따옴표로 감싼 주소에
    백슬래시까지 붙는 실수가 잦으므로 그냥 지우고 진행한다.
    """
    cleaned = url.strip().replace("\\", "")
    return cleaned, cleaned != url.strip()


def url_problem(url: str) -> str | None:
    """주소를 쓸 수 있으면 None, 아니면 사용자에게 보여줄 오류 메시지."""
    parsed = urlsplit(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return f"http(s) 주소가 아닙니다: {url}"
    return None


def print_presets() -> None:
    print("프리셋 (조각 크기는 '권장 폭' 기준으로 계산한 값):\n")
    for preset in PRESETS.values():
        width = preset.ideal_width or 1200
        height = min(preset.max_edge, preset.max_pixels // width)
        marker = "" if preset.ideal_width else " (원본 폭 유지, 1200 가정)"
        print(f"  {preset.name:13s} 긴 변 {preset.max_edge:>5,}px · 최대 {preset.max_pixels/1e6:.2f}MP")
        print(f"  {'':13s} 조각 {width}x{height}{marker} · 약 {preset.tokens(width, height):,} 토큰")
        print(f"  {'':13s} {preset.note}\n")
    print("공식 문서로 확인된 값은 claude 계열과 openai 뿐입니다.")
    print("gemini 는 커뮤니티 역공학 기반이라 --max-edge / --max-pixels 로 직접 조정하세요.")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.list_presets:
        print_presets()
        return 0

    if args.file and args.images:
        print("--file 은 이미지/URL 인자 없이 단독으로 씁니다.", file=sys.stderr)
        return 2

    targets: list[str | Path] = [
        arg if is_url(arg) else Path(arg) for arg in args.images
    ]
    if args.file:
        picked = pick_image_interactively()
        if picked is None:
            return 1
        targets = [picked]
    elif not targets:
        print("이미지 경로나 URL을 지정하세요 (또는 --file 로 선택창을 여세요).", file=sys.stderr)
        return 2

    if args.outdir and len(targets) > 1:
        print("--outdir 는 대상이 하나일 때만 쓸 수 있습니다.", file=sys.stderr)
        return 2

    status = 0
    for target in targets:
        if isinstance(target, str):
            status |= slice_url(target, args)
        elif target.is_file():
            status |= slice_image(target, args)
        else:
            print(f"파일을 찾을 수 없습니다: {target}", file=sys.stderr)
            status = 1
    return status


if __name__ == "__main__":
    sys.exit(main())
