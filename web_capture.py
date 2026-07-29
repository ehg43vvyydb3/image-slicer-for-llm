"""URL을 브라우저로 열어 스크롤하면서 화면 단위로 캡처한다.

브라우저의 '전체 페이지 스크린샷'은 세로 길이·총 면적에 한계가 있어서 긴 페이지는
잘리거나 실패한다. 어차피 잘라 쓸 거라면 한 장으로 찍을 이유가 없으므로, 뷰포트를
조각 크기에 맞춰 놓고 스크롤하며 찍는다. 그러면

  - 길이 제한이 사라지고 (잘림 없음),
  - 스크롤하는 과정에서 lazy loading 이미지가 전부 로딩되고,
  - 중간 거대 이미지 없이 바로 최종 조각이 나온다.
"""

from __future__ import annotations

import io
import math
import re
import time
from dataclasses import dataclass, field
from urllib.parse import urlsplit

from PIL import Image

from slice_for_llm import best_blank_row, row_variance

# 페이지 전체 높이 (브라우저마다 값이 달라서 최대값을 쓴다)
PAGE_HEIGHT_JS = """() => Math.max(
    document.body ? document.body.scrollHeight : 0,
    document.body ? document.body.offsetHeight : 0,
    document.documentElement.scrollHeight,
    document.documentElement.offsetHeight
)"""

SCROLL_Y_JS = "() => window.scrollY || document.documentElement.scrollTop || 0"

# lazy loading 을 강제로 풀어준다. 스크롤만으로 안 뜨는 이미지를 위한 안전장치.
FORCE_LAZY_JS = """() => {
    for (const img of document.images) {
        img.loading = 'eager';
        for (const attr of ['data-src', 'data-lazy-src', 'data-original', 'data-echo']) {
            const value = img.getAttribute(attr);
            if (value && !img.src.endsWith(value)) img.src = value;
        }
        const srcset = img.getAttribute('data-srcset');
        if (srcset && !img.srcset) img.srcset = srcset;
    }
    return document.images.length;
}"""

IMAGES_READY_JS = """() => Array.from(document.images)
    .every(img => img.complete || img.naturalWidth > 0)"""

# 고정/스티키 요소는 모든 조각에 겹쳐 나오므로 캡처 전에 숨긴다.
HIDE_STICKY_JS = """() => {
    let hidden = 0;
    for (const el of document.querySelectorAll('body *')) {
        const style = getComputedStyle(el);
        if ((style.position === 'fixed' || style.position === 'sticky')
            && style.visibility !== 'hidden' && style.display !== 'none') {
            el.style.setProperty('visibility', 'hidden', 'important');
            hidden += 1;
        }
    }
    return hidden;
}"""


@dataclass
class CapturedSlice:
    image: Image.Image
    page_top: int  # 페이지 좌표계(CSS 픽셀) 기준 위/아래 위치
    page_bottom: int


@dataclass
class CaptureResult:
    url: str
    final_url: str
    title: str
    page_height: int
    viewport: tuple[int, int]
    device_scale_factor: int
    slices: list[CapturedSlice] = field(default_factory=list)


class CaptureError(RuntimeError):
    pass


def slugify(text: str, fallback: str = "capture", limit: int = 60) -> str:
    """페이지 제목을 파일 이름으로 쓸 수 있게 다듬는다 (한글은 유지)."""
    text = re.sub(r"\s+", "_", text.strip())
    text = re.sub(r"[^0-9A-Za-z가-힣._-]", "", text)
    text = re.sub(r"_{2,}", "_", text).strip("._-")
    return text[:limit] or fallback


def plan_scroll_positions(page_height: int, tile_height: int, overlap: int) -> list[int]:
    """캡처할 스크롤 위치를 균등하게 배분한다.

    마지막에 몇십 픽셀짜리 자투리 조각이 생기지 않도록, 필요한 조각 수를 먼저 구한 뒤
    간격을 고르게 나눈다. 실제 겹침은 요청한 값 이상이 된다.
    """
    if page_height <= tile_height:
        return [0]
    span = page_height - tile_height
    step = max(1, tile_height - overlap)
    count = math.ceil(span / step) + 1
    return [round(i * span / (count - 1)) for i in range(count)]


def _require_playwright():
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:  # 로컬 파일만 자를 때는 playwright 가 없어도 된다
        raise CaptureError(
            "URL 캡처에는 playwright 가 필요합니다:\n"
            "  .venv/bin/python -m pip install playwright\n"
            "  .venv/bin/playwright install chromium"
        ) from exc
    return sync_playwright


def _find_content_frame_url(page) -> str | None:
    """최상위 페이지가 사실상 비어있을 때, 진짜 콘텐츠가 든 iframe 주소를 찾는다.

    네이버 블로그처럼 최상위 페이지는 빈 껍데기고 글 내용은 iframe(관례상
    mainFrame) 안의 별도 주소에 들어있는 사이트를 위한 보정이다. 최상위 body에
    이미 내용이 있으면 그대로 두고 None을 돌려준다.
    """
    body_len = page.evaluate(
        "() => (document.body ? document.body.innerText : '').trim().length"
    )
    if body_len > 100:
        return None

    candidates = [
        f for f in page.frames
        if f != page.main_frame and f.url and f.url != "about:blank"
    ]
    if not candidates:
        return None

    for f in candidates:
        if f.name == "mainFrame":
            return f.url

    host = urlsplit(page.url).hostname or ""
    for f in candidates:
        fhost = urlsplit(f.url).hostname or ""
        if fhost and (fhost == host or fhost.endswith("." + host.split(".", 1)[-1])):
            return f.url

    return None


def _scroll_through_page(page, step: int, log) -> int:
    """끝까지 훑어서 lazy loading 을 유발하고 최종 페이지 높이를 돌려준다."""
    height = page.evaluate(PAGE_HEIGHT_JS)
    position = 0
    for _ in range(500):  # 무한 스크롤 페이지에서 영원히 돌지 않도록
        position += step
        page.evaluate("y => window.scrollTo(0, y)", position)
        page.wait_for_timeout(120)
        new_height = page.evaluate(PAGE_HEIGHT_JS)
        if position >= new_height:
            if new_height <= height:
                height = new_height
                break
        height = new_height
    else:
        log("  주의: 페이지가 계속 길어져서 스크롤을 중간에 멈췄습니다 (무한 스크롤?)")
    return height


def _wait_for_images(page, timeout_ms: int) -> None:
    page.evaluate(FORCE_LAZY_JS)
    deadline = time.monotonic() + timeout_ms / 1000
    while time.monotonic() < deadline:
        if page.evaluate(IMAGES_READY_JS):
            return
        page.wait_for_timeout(200)


def _fuzzy_text_pattern(text: str) -> re.Pattern[str]:
    """글자 사이 공백 유무를 무시하는 정규식을 만든다.

    '더보기'로 찾아도 실제 버튼 텍스트가 '더 보기'(또는 그 반대)면 그냥 substring
    매칭으로는 못 찾는다. 공백을 걷어낸 글자들 사이에 \\s* 를 끼워 넣어 양쪽 다
    잡히게 한다.
    """
    chars = [c for c in text if not c.isspace()]
    pattern = r"\s*".join(re.escape(c) for c in chars)
    return re.compile(pattern, re.IGNORECASE)


def _count_with_wait(page, locator, timeout_ms: int = 10000, poll_ms: int = 250) -> int:
    """count() 는 그 순간의 DOM만 보고 기다려주지 않는다.

    광고/리뷰 위젯이 비동기로 늦게 뜨는 페이지에서는 스크롤이 끝난 시점에도 아직
    버튼이 안 붙어 있을 수 있으므로, 하나라도 잡힐 때까지 잠깐 폴링한다.
    """
    deadline = time.monotonic() + timeout_ms / 1000
    while True:
        count = locator.count()
        if count > 0 or time.monotonic() >= deadline:
            return count
        page.wait_for_timeout(poll_ms)


def _click_expand_texts(page, texts: list[str], log, max_rounds: int = 20) -> None:
    """지정한 텍스트를 가진 요소를 모두 클릭한다 ('더보기' 류 펼치기 버튼용).

    클릭할 때마다 버튼이 사라지거나(펼쳐진 뒤 없어짐) 아래쪽에 같은 텍스트의 새
    버튼이 나타날 수 있어서, 더 이상 찾을 게 없을 때까지 매번 다시 찾아서 누른다.
    """
    for text in texts:
        clicked_total = 0
        found_ever = False
        pattern = _fuzzy_text_pattern(text)
        for _ in range(max_rounds):
            locator = page.get_by_text(pattern)
            count = _count_with_wait(page, locator)
            if count == 0:
                break
            found_ever = True
            clicked_this_round = 0
            for i in range(count):
                element = locator.nth(i)
                try:
                    if not element.is_visible():
                        continue
                    element.scroll_into_view_if_needed(timeout=2000)
                    element.click(timeout=2000)
                except Exception:
                    continue
                clicked_this_round += 1
                clicked_total += 1
                page.wait_for_timeout(300)
            if clicked_this_round == 0:
                break
        if clicked_total:
            log(f"  '{text}' {clicked_total}번 클릭")
        elif found_ever:
            log(f"  '{text}' 요소는 찾았지만 클릭이 막혔습니다 (팝업/오버레이가 가리고 있을 수 있음)")
        else:
            log(f"  '{text}' 텍스트를 가진 요소를 찾지 못했습니다")


def capture_slices(
    url: str,
    *,
    width: int,
    tile_height: int,
    overlap: int,
    scale: int = 2,
    smart_window: int = 80,
    smart_cut: bool = True,
    engine: str = "chromium",
    headless: bool = True,
    extra_wait: float = 0.0,
    timeout: float = 60.0,
    click_texts: list[str] | None = None,
    pause: bool = False,
    log=print,
) -> CaptureResult:
    sync_playwright = _require_playwright()
    timeout_ms = int(timeout * 1000)
    if pause and headless:
        raise CaptureError("pause 는 브라우저 창이 보여야 쓸 수 있습니다 (headless=False 필요)")

    with sync_playwright() as pw:
        launcher = getattr(pw, engine, None)
        if launcher is None:
            raise CaptureError(f"알 수 없는 엔진입니다: {engine}")
        try:
            browser = launcher.launch(headless=headless)
        except Exception as exc:
            raise CaptureError(
                f"{engine} 브라우저를 실행하지 못했습니다. 설치가 필요할 수 있습니다:\n"
                f"  .venv/bin/playwright install {engine}\n원본 오류: {exc}"
            ) from exc

        try:
            context = browser.new_context(
                viewport={"width": width, "height": tile_height},
                device_scale_factor=scale,
            )
            page = context.new_page()
            page.set_default_timeout(timeout_ms)

            log(f"  페이지 여는 중: {url}")
            response = page.goto(url, wait_until="load", timeout=timeout_ms)

            status = response.status if response else None
            if status and status >= 400:
                log(f"  주의: 서버가 HTTP {status} 를 반환했습니다.")
            # 오류 페이지로 리다이렉트되는 경우가 많다 (주소 오타, 로그인 요구 등).
            # 상태 코드는 200 이어도 주소가 바뀌므로 이쪽이 더 확실한 신호다.
            if page.url != url:
                log(f"  이동됨 → {page.url}")

            frame_url = _find_content_frame_url(page)
            if frame_url:
                log(f"  본문이 빈 껍데기 안 iframe에 있어서 실제 주소로 다시 엽니다: {frame_url}")
                response = page.goto(frame_url, wait_until="load", timeout=timeout_ms)
                status = response.status if response else None
                if status and status >= 400:
                    log(f"  주의: 서버가 HTTP {status} 를 반환했습니다.")

            log("  끝까지 스크롤해서 이미지 로딩 중...")
            page_height = _scroll_through_page(page, int(tile_height * 0.8), log)
            _wait_for_images(page, timeout_ms)
            try:
                page.wait_for_load_state("networkidle", timeout=5000)
            except Exception:
                pass  # 광고·트래커가 계속 붙는 페이지에서는 networkidle 이 안 온다
            if extra_wait:
                page.wait_for_timeout(int(extra_wait * 1000))

            if click_texts:
                log("  펼치기 버튼 클릭 중...")
                _click_expand_texts(page, click_texts, log)
                # 클릭으로 새 내용이 펼쳐졌을 수 있으니 다시 끝까지 스크롤해서 로딩시킨다.
                page_height = _scroll_through_page(page, int(tile_height * 0.8), log)
                _wait_for_images(page, timeout_ms)

            if pause:
                # 봇 탐지로 자동 클릭이 막히는 페이지(알리익스프레스 등)를 위한
                # 사람 개입 지점. 더보기 클릭·로그인·캡차를 직접 처리하게 한다.
                log("  브라우저 창에서 필요한 조작(더보기 클릭, 로그인 등)을 직접 하세요.")
                input("  준비되면 엔터를 누르세요 (캡처를 시작합니다)... ")
                # 사람이 펼친 내용까지 마저 로딩시킨다.
                page_height = _scroll_through_page(page, int(tile_height * 0.8), log)
                _wait_for_images(page, timeout_ms)

            hidden = page.evaluate(HIDE_STICKY_JS)
            if hidden:
                log(f"  고정/스티키 요소 {hidden}개 숨김 (조각마다 겹쳐 나오는 것 방지)")

            page.evaluate("() => window.scrollTo(0, 0)")
            page.wait_for_timeout(300)
            page_height = page.evaluate(PAGE_HEIGHT_JS)
            title = page.title() or ""
            log(f"  페이지 높이 {page_height:,}px · 뷰포트 {width}x{tile_height} @{scale}x")

            result = CaptureResult(
                url=url,
                final_url=page.url,
                title=title,
                page_height=page_height,
                viewport=(width, tile_height),
                device_scale_factor=scale,
            )

            positions = plan_scroll_positions(page_height, tile_height, overlap)
            log(f"  조각 {len(positions)}개 캡처 중...")

            for i, requested in enumerate(positions):
                page.evaluate("y => window.scrollTo(0, y)", requested)
                page.wait_for_timeout(150)
                actual = page.evaluate(SCROLL_Y_JS)

                shot = Image.open(io.BytesIO(page.screenshot(animations="disabled"))).convert("RGB")
                if shot.size != (width, tile_height):
                    shot = shot.resize((width, tile_height), Image.LANCZOS)

                # 페이지가 뷰포트보다 짧으면 아래쪽 빈 공간을 잘라낸다.
                bottom = min(tile_height, max(1, page_height - actual))

                if smart_cut and i + 1 < len(positions):
                    # 다음 조각 시작점보다 위로는 자를 수 없고(빈 구간 방지),
                    # 문맥이 이어지도록 요청 겹침의 절반은 남긴다.
                    floor = max(
                        tile_height - smart_window,
                        positions[i + 1] - actual + overlap // 2,
                    )
                    bottom = best_blank_row(row_variance(shot), bottom, floor)

                if bottom < tile_height:
                    shot = shot.crop((0, 0, width, bottom))
                result.slices.append(CapturedSlice(shot, actual, actual + shot.height))

            return result
        finally:
            browser.close()
