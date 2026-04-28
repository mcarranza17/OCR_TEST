from __future__ import annotations

import base64
import io
import json
import os
import socket
from pathlib import Path

import streamlit as st

from src.dni_parser import parse_honduras_dni
from src.face_matcher import FaceMatcher, FaceMatcherError
from src.image_io import image_bytes_to_bgr, normalize_upright_jpeg
from src.ocr import OCRError, get_paddle_ocr, run_ocr_with
from src.session_store import (
    cleanup_old,
    create_session,
    read_image,
    read_state,
    reset_kind,
    save_image,
)
from src.settings import Settings


LOGO_PATH = Path(__file__).parent / "gladiium.png"
UPLOAD_TYPES = ["jpg", "jpeg", "png", "heic", "heif", "webp"]


def _public_base_url() -> str:
    forced = os.getenv("PUBLIC_URL")
    if forced:
        return str(forced).rstrip("/")
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.connect(("8.8.8.8", 80))
        ip = sock.getsockname()[0]
        sock.close()
    except Exception:  # noqa: BLE001 - best-effort detection.
        ip = "localhost"
    port = os.getenv("STREAMLIT_SERVER_PORT", "8501")
    return f"http://{ip}:{port}"


def _make_qr_png(url: str) -> bytes:
    import qrcode

    qr = qrcode.QRCode(box_size=8, border=2)
    qr.add_data(url)
    qr.make()
    img = qr.make_image(fill_color="#0A0A0A", back_color="#FFFFFF")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _read_image_bytes(uploaded) -> bytes:
    """Return upright JPEG-normalized bytes from an upload or camera capture.

    Always applies EXIF orientation and re-encodes as JPEG. Fixes:
    - iPhone uploads stored sideways with a rotation tag.
    - HEIC/HEIF inputs (iOS default).
    - Inconsistencies between st.image preview and downstream OCR/face matcher.
    """
    return normalize_upright_jpeg(uploaded.getvalue())


LIGHT_VARS = {
    "--bg": "#FFFFFF",
    "--surface": "#FAF7F5",
    "--surface-2": "#F2EDEA",
    "--text": "#0A0A0A",
    "--muted": "#6B6B6B",
    "--border": "rgba(10,10,10,0.10)",
    "--border-strong": "rgba(122,20,20,0.35)",
    "--accent": "#7A1414",
    "--accent-2": "#4A0A0A",
    "--accent-soft": "rgba(122,20,20,0.06)",
    "--ok": "#14784A",
    "--warn": "#7A1414",
    "--bad": "#B00020",
    "--logo-bg": "transparent",
    "--shadow": "0 6px 20px rgba(10,10,10,0.06)",
    "--btn-text": "#FFFFFF",
}

DARK_VARS = {
    "--bg": "#0D0B0B",
    "--surface": "#161212",
    "--surface-2": "#1F1818",
    "--text": "#F2EEEC",
    "--muted": "#8E8884",
    "--border": "rgba(255,255,255,0.10)",
    "--border-strong": "rgba(196,69,69,0.55)",
    "--accent": "#C44545",
    "--accent-2": "#7A1414",
    "--accent-soft": "rgba(196,69,69,0.10)",
    "--ok": "#3DD68A",
    "--warn": "#C44545",
    "--bad": "#FF6B6B",
    "--logo-bg": "#FFFFFF",
    "--shadow": "0 6px 24px rgba(0,0,0,0.45)",
    "--btn-text": "#0D0B0B",
}


def _vars_block(vars_map: dict[str, str]) -> str:
    return "\n".join(f"  {k}: {v};" for k, v in vars_map.items())


def build_css(dark: bool) -> str:
    palette = DARK_VARS if dark else LIGHT_VARS
    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');

:root {{
{_vars_block(palette)}
}}

html, body, [class*="css"], .stApp, .stMarkdown, .stText, button, input, textarea, select {{
  font-family: 'Poppins', system-ui, -apple-system, sans-serif !important;
  color: var(--text);
}}

.stApp {{ background: var(--bg); }}

header[data-testid="stHeader"] {{ background: transparent; }}
#MainMenu, footer {{ visibility: hidden; }}

.block-container {{
  padding-top: 0.6rem;
  padding-bottom: 2rem;
  max-width: 1080px;
}}

/* Brand bar */
.gld-brand {{
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 2px 0 12px 0;
  border-bottom: 1px solid var(--border);
  margin-bottom: 18px;
  gap: 12px;
}}
.gld-logo-wrap {{
  background: var(--logo-bg);
  padding: {"6px 10px" if dark else "0"};
  border-radius: 8px;
  display: inline-flex;
  align-items: center;
}}
.gld-logo-wrap img {{
  height: 32px;
  width: auto;
  display: block;
}}
.gld-pill {{
  font-size: 10px;
  letter-spacing: 0.2em;
  text-transform: uppercase;
  color: var(--accent);
  border: 1px solid var(--border-strong);
  padding: 4px 10px;
  border-radius: 999px;
  background: var(--accent-soft);
  font-weight: 600;
}}

/* Hero */
.gld-hero {{ margin-bottom: 4px; }}
.gld-hero h1 {{
  font-family: 'Poppins', sans-serif !important;
  font-weight: 700;
  letter-spacing: -0.015em;
  font-size: 24px;
  line-height: 1.15;
  margin: 2px 0 2px 0;
  color: var(--text);
}}
.gld-hero h1 em {{
  font-style: normal;
  color: var(--accent);
}}
.gld-hero p {{
  color: var(--muted);
  margin: 0;
  font-size: 13px;
  font-weight: 400;
}}

/* Section labels */
.gld-step {{
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 4px 10px;
  border: 1px solid var(--border);
  border-radius: 999px;
  font-size: 10px;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  color: var(--muted);
  background: transparent;
  margin: 18px 0 8px 0;
  font-weight: 600;
}}
.gld-step b {{
  color: var(--accent);
  font-weight: 700;
}}

/* Mobile step header */
.gld-mstep {{
  display: flex;
  align-items: center;
  gap: 10px;
  margin: 10px 0 12px;
}}
.gld-mstep-num {{
  width: 28px; height: 28px;
  border-radius: 50%;
  background: var(--accent);
  color: #FFFFFF;
  display: grid; place-items: center;
  font-weight: 700; font-size: 13px;
  flex-shrink: 0;
}}
.gld-mstep h3 {{
  margin: 0;
  font-size: 15px;
  font-weight: 600;
  letter-spacing: 0;
  text-transform: none;
  color: var(--text);
}}
.gld-mstep .gld-mstep-sub {{
  display: block;
  font-size: 12px;
  color: var(--muted);
  font-weight: 400;
  margin-top: 2px;
}}

/* Visual frame: DNI rectangle */
.gld-doc-frame {{
  position: relative;
  aspect-ratio: 1.585;
  border: 1px dashed var(--border-strong);
  border-radius: 12px;
  background: var(--accent-soft);
  margin: 0 auto 22px;
  max-width: 320px;
}}
.gld-corner {{
  position: absolute;
  width: 18px; height: 18px;
  border: 3px solid var(--accent);
}}
.gld-corner.tl {{ top: 8px; left: 8px; border-right: none; border-bottom: none; border-radius: 4px 0 0 0; }}
.gld-corner.tr {{ top: 8px; right: 8px; border-left: none; border-bottom: none; border-radius: 0 4px 0 0; }}
.gld-corner.bl {{ bottom: 8px; left: 8px; border-right: none; border-top: none; border-radius: 0 0 0 4px; }}
.gld-corner.br {{ bottom: 8px; right: 8px; border-left: none; border-top: none; border-radius: 0 0 4px 0; }}
.gld-frame-label {{
  position: absolute;
  bottom: -10px; left: 50%; transform: translateX(-50%);
  font-size: 10px;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  color: var(--accent);
  background: var(--bg);
  padding: 2px 10px;
  border-radius: 4px;
  font-weight: 600;
  white-space: nowrap;
}}

/* Visual frame: selfie oval */
.gld-face-frame {{
  width: 140px;
  height: 180px;
  margin: 6px auto 28px;
  display: block;
  border: 2px dashed var(--accent);
  border-radius: 50%;
  background: var(--accent-soft);
  position: relative;
}}
.gld-face-frame::after {{
  content: "Centra tu rostro";
  position: absolute;
  bottom: -22px; left: 50%; transform: translateX(-50%);
  font-size: 10px;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: var(--accent);
  font-weight: 600;
  white-space: nowrap;
}}

/* Checklist with custom checkbox icon */
.gld-checklist {{
  list-style: none;
  padding: 0;
  margin: 14px 0;
  font-size: 13px;
  color: var(--text);
  display: grid;
  gap: 6px;
}}
.gld-checklist li {{
  position: relative;
  padding: 4px 0 4px 24px;
  line-height: 1.35;
}}
.gld-checklist li::before {{
  content: "";
  position: absolute;
  left: 0; top: 7px;
  width: 14px; height: 14px;
  border-radius: 4px;
  background: var(--accent-soft);
  border: 1px solid var(--accent);
}}
.gld-checklist li::after {{
  content: "";
  position: absolute;
  left: 4px; top: 10px;
  width: 6px; height: 3px;
  border-left: 2px solid var(--accent);
  border-bottom: 2px solid var(--accent);
  transform: rotate(-45deg);
}}

/* Progress bar */
.gld-progress {{
  margin: 0 0 14px;
}}
.gld-progress-bar {{
  height: 4px;
  background: var(--surface-2);
  border-radius: 2px;
  overflow: hidden;
}}
.gld-progress-bar > div {{
  height: 100%;
  background: var(--accent);
  transition: width .25s ease;
}}
.gld-progress-label {{
  font-size: 11px;
  color: var(--muted);
  margin-top: 6px;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  font-weight: 600;
}}

/* Live overlay over st.camera_input video — only while video is active.
   Uses :has() (supported in iOS Safari 15.4+, Chrome 105+, Firefox 121+). */
[data-testid="stCameraInput"]:has(video) {{
  position: relative;
  overflow: hidden;
  border-radius: 12px;
}}

/* DNI step → wide rectangle guide */
[data-testid="stVerticalBlock"]:has(.gld-cam-mark-doc) [data-testid="stCameraInput"]:has(video)::after {{
  content: "";
  position: absolute;
  top: 50%; left: 50%;
  transform: translate(-50%, -50%);
  width: 92%;
  aspect-ratio: 1.585;
  border: 2px dashed rgba(255, 255, 255, 0.92);
  border-radius: 8px;
  pointer-events: none;
  z-index: 10;
  box-shadow: 0 0 0 2000px rgba(0, 0, 0, 0.40);
}}

/* Selfie step → oval face guide */
[data-testid="stVerticalBlock"]:has(.gld-cam-mark-self) [data-testid="stCameraInput"]:has(video)::after {{
  content: "";
  position: absolute;
  top: 50%; left: 50%;
  transform: translate(-50%, -50%);
  width: 56%;
  aspect-ratio: 0.78;
  border: 3px dashed rgba(255, 255, 255, 0.95);
  border-radius: 50%;
  pointer-events: none;
  z-index: 10;
  box-shadow: 0 0 0 2000px rgba(0, 0, 0, 0.42);
}}

/* Capture card */
.gld-card {{
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 12px 14px;
  box-shadow: none;
}}
.gld-card h3 {{
  font-family: 'Poppins', sans-serif;
  font-weight: 600;
  letter-spacing: 0;
  color: var(--text);
  margin: 0 0 8px 0;
  font-size: 13px;
  text-transform: uppercase;
  letter-spacing: 0.06em;
}}
.gld-card .gld-hint {{
  color: var(--muted);
  font-size: 11px;
  margin-top: 8px;
  border-top: 1px solid var(--border);
  padding-top: 6px;
  font-weight: 400;
  letter-spacing: 0.02em;
}}

/* Camera + uploader */
[data-testid="stCameraInput"] video,
[data-testid="stCameraInput"] img {{
  border-radius: 12px;
  border: 1px solid var(--border);
}}
[data-testid="stCameraInput"] button {{
  background: var(--text) !important;
  color: var(--bg) !important;
  border: none !important;
  border-radius: 8px !important;
  font-weight: 500 !important;
}}
[data-testid="stFileUploader"] section {{
  border: 1px dashed var(--border-strong);
  border-radius: 12px;
  background: var(--accent-soft);
}}
[data-testid="stFileUploader"] * {{ color: var(--text); }}

/* Buttons */
.stButton > button, .stDownloadButton > button {{
  background: var(--accent);
  color: #FFFFFF;
  border: none;
  border-radius: 8px;
  font-weight: 600;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  font-size: 12px;
  padding: 9px 18px;
  box-shadow: none;
  transition: filter .15s ease;
  font-family: 'Poppins', sans-serif !important;
}}
.stButton > button:hover {{
  filter: brightness(1.08);
}}

/* Radio */
[role="radiogroup"] label {{
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 5px 11px;
  margin-right: 6px;
  background: var(--surface);
  color: var(--text) !important;
  font-size: 12px;
}}
[role="radiogroup"] label[data-checked="true"] {{
  border-color: var(--accent);
  background: var(--accent-soft);
}}

/* Sliders */
[data-baseweb="slider"] div[role="slider"] {{
  background: var(--accent) !important;
  border-color: var(--accent) !important;
}}

/* Decision banners */
.gld-banner {{
  border-radius: 10px;
  padding: 12px 16px;
  margin: 4px 0 14px 0;
  border: 1px solid var(--border);
  display: flex; align-items: center; gap: 10px;
  font-weight: 600; letter-spacing: 0.12em;
  text-transform: uppercase;
  font-size: 12px;
  background: var(--surface);
}}
.gld-banner .dot {{
  width: 8px; height: 8px; border-radius: 50%;
}}
.gld-verified  {{ color: var(--ok);   border-color: color-mix(in srgb, var(--ok) 35%, transparent); }}
.gld-review    {{ color: var(--warn); border-color: var(--border-strong); }}
.gld-rejected  {{ color: var(--bad);  border-color: color-mix(in srgb, var(--bad) 40%, transparent); }}

/* Metrics / JSON / Expander */
[data-testid="stMetricValue"] {{
  color: var(--accent);
  font-family: 'Poppins', sans-serif;
  font-weight: 700;
}}
[data-testid="stMetricLabel"] {{ color: var(--muted); }}
[data-testid="stExpander"] {{
  border: 1px solid var(--border);
  border-radius: 12px;
  background: var(--surface);
}}
[data-testid="stExpander"] summary {{ color: var(--text); }}
[data-testid="stExpander"] [data-testid="stMarkdownContainer"] {{ color: var(--text); }}

/* Plain text output (st.text) — was rendering white-on-white in light mode */
[data-testid="stText"],
[data-testid="stText"] * {{
  color: var(--text) !important;
  font-family: 'Poppins', monospace !important;
}}
[data-testid="stText"] {{
  background: var(--surface-2) !important;
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 12px 14px !important;
  font-size: 13px !important;
  line-height: 1.5 !important;
  white-space: pre-wrap;
  word-break: break-word;
}}

/* Code blocks (st.code) */
[data-testid="stCode"],
[data-testid="stCodeBlock"],
.stCodeBlock,
pre, code {{
  color: var(--text) !important;
}}
[data-testid="stCode"] pre,
[data-testid="stCodeBlock"] pre,
pre {{
  background: var(--surface-2) !important;
  border: 1px solid var(--border);
  border-radius: 10px;
}}
[data-testid="stCode"] code,
[data-testid="stCodeBlock"] code,
pre code {{
  background: transparent !important;
  color: var(--text) !important;
}}
[data-testid="stCode"] button,
[data-testid="stCodeBlock"] button {{
  color: var(--text) !important;
  background: transparent !important;
}}

/* JSON viewer */
[data-testid="stJson"] {{
  background: var(--surface-2) !important;
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 8px 10px;
}}
[data-testid="stJson"] * {{
  color: var(--text) !important;
}}
[data-testid="stJson"] .key,
[data-testid="stJson"] [class*="key"] {{
  color: var(--accent) !important;
}}

/* Sidebar */
[data-testid="stSidebar"] {{
  background: var(--surface);
  border-right: 1px solid var(--border);
}}
[data-testid="stSidebar"] * {{ color: var(--text); }}
[data-testid="stSidebar"] h2 {{
  font-family: 'Poppins', sans-serif;
  font-weight: 700;
  letter-spacing: 0.04em;
  color: var(--accent);
  font-size: 14px;
  text-transform: uppercase;
}}

/* Toggle */
[data-testid="stToggle"] label {{ color: var(--text); }}

/* Info / alerts */
[data-baseweb="notification"] {{
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: 12px !important;
  color: var(--text) !important;
}}
</style>
"""


def _logo_data_uri() -> str | None:
    if not LOGO_PATH.exists():
        return None
    encoded = base64.b64encode(LOGO_PATH.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def render_brand_header() -> None:
    logo_uri = _logo_data_uri()
    logo_html = (
        f'<div class="gld-logo-wrap"><img src="{logo_uri}" alt="Gladiium Technology Partners" /></div>'
        if logo_uri
        else '<div style="font-family:Poppins,sans-serif;font-weight:700;font-size:22px;color:var(--accent);">GLADIIUM</div>'
    )
    st.markdown(
        f"""
        <div class="gld-brand">
          {logo_html}
          <div class="gld-pill">Identity Verification · Honduras DNI</div>
        </div>
        <div class="gld-hero">
          <h1>Verificación de <em>Identidad</em></h1>
          <p>OCR documental y comparación biométrica del DNI con selfie.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_mobile_progress(doc_ready: bool, selfie_ready: bool) -> None:
    done = sum([doc_ready, selfie_ready])
    pct = int(done / 2 * 100)
    st.markdown(
        f"""
        <div class="gld-progress">
          <div class="gld-progress-bar"><div style="width:{pct}%"></div></div>
          <div class="gld-progress-label">{done} de 2 capturas listas</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_step_header(num: int, title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="gld-mstep">
          <div class="gld-mstep-num">{num}</div>
          <div>
            <h3>{title}</h3>
            <span class="gld-mstep-sub">{subtitle}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_doc_frame() -> None:
    st.markdown(
        """
        <div class="gld-doc-frame">
          <span class="gld-corner tl"></span>
          <span class="gld-corner tr"></span>
          <span class="gld-corner bl"></span>
          <span class="gld-corner br"></span>
          <span class="gld-frame-label">Encuadrar el DNI completo</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_face_frame() -> None:
    st.markdown('<div class="gld-face-frame"></div>', unsafe_allow_html=True)


def _render_checklist(items: list[str]) -> None:
    lis = "".join(f"<li>{item}</li>" for item in items)
    st.markdown(f'<ul class="gld-checklist">{lis}</ul>', unsafe_allow_html=True)


def render_mobile_capture(sid: str) -> None:
    state = read_state(sid)
    if state is None:
        st.error("Sesión inválida o expirada. Pide al desktop que genere un QR nuevo.")
        return

    doc_ready = state.get("document") == "ready"
    selfie_ready = state.get("selfie") == "ready"

    _render_mobile_progress(doc_ready, selfie_ready)

    if not doc_ready:
        _render_step_header(
            1,
            "Foto del DNI",
            "Gira el celular horizontal ↔ y encuadra el DNI completo dentro del marco.",
        )
        _render_doc_frame()
        _render_checklist([
            "Lugar bien iluminado, sin contraluz",
            "Sin reflejos sobre la mica del DNI",
            "Las 4 esquinas dentro del marco",
            "Texto y foto se leen claramente",
        ])
        st.markdown('<span class="gld-cam-mark-doc"></span>', unsafe_allow_html=True)
        photo = st.camera_input("Tomar foto del DNI", key="m_doc_cam")
        with st.expander("O subir desde galería"):
            upload = st.file_uploader(
                "Subir desde galería",
                type=UPLOAD_TYPES,
                key="m_doc_upl",
                label_visibility="collapsed",
            )
        chosen = photo or upload
        if chosen is not None:
            try:
                normalized = _read_image_bytes(chosen)
            except ValueError as exc:
                st.error(str(exc))
                return
            save_image(sid, "document", normalized)
            st.rerun()
        return

    if not selfie_ready:
        _render_step_header(
            2,
            "Selfie de tu rostro",
            "Sostén el celular vertical y centra tu rostro en el círculo.",
        )
        _render_face_frame()
        _render_checklist([
            "Buena luz frontal, no a contraluz",
            "Sin gorra ni lentes oscuros",
            "Rostro completo, ojos abiertos",
            "Expresión neutral, mira a la cámara",
        ])
        st.markdown('<span class="gld-cam-mark-self"></span>', unsafe_allow_html=True)
        photo = st.camera_input("Tomar selfie", key="m_self_cam")
        with st.expander("O subir desde galería"):
            upload = st.file_uploader(
                "Subir desde galería",
                type=UPLOAD_TYPES,
                key="m_self_upl",
                label_visibility="collapsed",
            )
        chosen = photo or upload
        if chosen is not None:
            try:
                normalized = _read_image_bytes(chosen)
            except ValueError as exc:
                st.error(str(exc))
                return
            save_image(sid, "selfie", normalized)
            st.rerun()
        if st.button("Volver a tomar el DNI", key="m_redo_doc"):
            reset_kind(sid, "document")
            st.rerun()
        return

    st.markdown(
        """
        <div class="gld-banner gld-verified">
          <span class="dot" style="background: currentColor;"></span>
          <span>Listo. Vuelve a la computadora para ver el resultado.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    cols = st.columns(2)
    if cols[0].button("Re-tomar DNI"):
        reset_kind(sid, "document")
        st.rerun()
    if cols[1].button("Re-tomar selfie"):
        reset_kind(sid, "selfie")
        st.rerun()


def render_qr_handoff() -> tuple[bytes | None, bytes | None]:
    if "qr_sid" not in st.session_state:
        st.session_state["qr_sid"] = create_session()
    sid = st.session_state["qr_sid"]

    state = read_state(sid)
    if state is None:
        st.session_state["qr_sid"] = create_session()
        sid = st.session_state["qr_sid"]
        state = read_state(sid) or {}

    base_url = _public_base_url()
    mobile_url = f"{base_url}/?mobile=1&sid={sid}"

    qr_col, info_col = st.columns([1, 2], gap="large")
    with qr_col:
        st.image(_make_qr_png(mobile_url), width=240)
    with info_col:
        st.markdown('<div class="gld-card"><h3>Captura desde tu teléfono</h3>', unsafe_allow_html=True)
        st.markdown(
            "1. Escanea el QR con la camara del telefono.<br>"
            "2. Toma la foto del DNI con buena luz.<br>"
            "3. Toma la selfie centrando el rostro.<br>"
            "4. Esta página se actualiza sola.",
            unsafe_allow_html=True,
        )
        st.code(mobile_url, language="text")
        c1, c2 = st.columns(2)
        if c1.button("Generar nuevo QR"):
            st.session_state.pop("qr_sid", None)
            st.rerun()
        if c2.button("Limpiar capturas"):
            reset_kind(sid, "document")
            reset_kind(sid, "selfie")
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    state = read_state(sid) or {}
    doc_ready = state.get("document") == "ready"
    selfie_ready = state.get("selfie") == "ready"

    status_col1, status_col2 = st.columns(2)
    with status_col1:
        st.markdown('<div class="gld-card"><h3>DNI</h3>', unsafe_allow_html=True)
        if doc_ready:
            doc_bytes = read_image(sid, "document")
            if doc_bytes:
                st.image(doc_bytes, use_column_width=True)
        else:
            st.caption("Pendiente — captura desde el teléfono.")
        st.markdown("</div>", unsafe_allow_html=True)
    with status_col2:
        st.markdown('<div class="gld-card"><h3>Selfie</h3>', unsafe_allow_html=True)
        if selfie_ready:
            selfie_bytes = read_image(sid, "selfie")
            if selfie_bytes:
                st.image(selfie_bytes, use_column_width=True)
        else:
            st.caption("Pendiente — captura desde el teléfono.")
        st.markdown("</div>", unsafe_allow_html=True)

    if not (doc_ready and selfie_ready):
        try:
            from streamlit_autorefresh import st_autorefresh

            st_autorefresh(interval=2000, key=f"qr_poll_{sid}")
        except ImportError:
            st.caption("streamlit-autorefresh no está instalado; refresca la página manualmente.")
        return None, None

    return read_image(sid, "document"), read_image(sid, "selfie")


def render_decision(decision: str) -> None:
    mapping = {
        "verified": ("gld-verified", "Identidad verificada"),
        "manual_review": ("gld-review", "Requiere revisión manual"),
        "rejected": ("gld-rejected", "Identidad rechazada"),
    }
    cls, label = mapping.get(decision, ("gld-review", decision))
    st.markdown(
        f"""
        <div class="gld-banner {cls}">
          <span class="dot" style="background: currentColor;"></span>
          <span>{label}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def decide(ocr_valid: bool, face_decision: str | None) -> str:
    if not ocr_valid or face_decision is None:
        return "manual_review"
    if face_decision == "match":
        return "verified"
    if face_decision == "review":
        return "manual_review"
    return "rejected"


@st.cache_resource(show_spinner=False)
def _cached_paddle_ocr(lang: str):
    return get_paddle_ocr(lang)


@st.cache_resource(show_spinner=False)
def _cached_face_matcher(model_name: str, model_root: str) -> FaceMatcher:
    # Built once per process; thresholds are mutated per call below.
    return FaceMatcher(model_name=model_name, model_root=model_root)


def _run_pipeline(
    settings: Settings,
    doc_bytes: bytes,
    selfie_bytes: bytes,
    face_match_threshold: float,
    face_review_threshold: float,
) -> None:
    progress = st.progress(0, text="Iniciando verificación…")

    try:
        progress.progress(10, text="Cargando modelos…")
        ocr_instance = _cached_paddle_ocr(settings.ocr_lang)
        matcher = _cached_face_matcher(settings.face_model_name, settings.insightface_root)
        matcher.match_threshold = face_match_threshold
        matcher.review_threshold = face_review_threshold

        progress.progress(30, text="Leyendo texto del DNI…")
        try:
            ocr_result = run_ocr_with(ocr_instance, doc_bytes)
        except OCRError as exc:
            progress.empty()
            msg = str(exc)
            if msg:
                st.error(msg)
            return

        progress.progress(55, text="Extrayendo campos…")
        parsed = parse_honduras_dni(ocr_result.full_text, ocr_result.average_confidence)

        progress.progress(70, text="Detectando rostros…")
        try:
            doc_bgr = image_bytes_to_bgr(doc_bytes)
            selfie_bgr = image_bytes_to_bgr(selfie_bytes)
        except ValueError as exc:
            progress.empty()
            st.error(str(exc))
            return

        progress.progress(85, text="Comparando biometría…")
        try:
            face_result = matcher.compare(doc_bgr, selfie_bgr)
        except (FaceMatcherError, ValueError) as exc:
            progress.empty()
            st.error(str(exc))
            return

        progress.progress(100, text="Listo")
    finally:
        progress.empty()

    final_decision = decide(parsed.is_valid_for_demo, face_result.decision)
    render_decision(final_decision)

    result_payload = {
        "decision": final_decision,
        "document": parsed.to_dict(),
        "face": face_result.to_dict(),
        "ocr": {
            "average_confidence": ocr_result.average_confidence,
            "lines": [line.to_dict() for line in ocr_result.lines],
        },
    }

    doc_col, face_col = st.columns(2, gap="large")
    with doc_col:
        st.markdown('<div class="gld-card"><h3>Campos extraidos</h3>', unsafe_allow_html=True)
        st.json(parsed.to_dict())
        st.markdown("</div>", unsafe_allow_html=True)

    with face_col:
        st.markdown('<div class="gld-card"><h3>Resultado biométrico</h3>', unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        m1.metric("Similitud", f"{face_result.similarity:.3f}")
        m2.metric("Rostro DNI", f"{face_result.document_face_ratio*100:.1f}%")
        m3.metric("Rostro selfie", f"{face_result.selfie_face_ratio*100:.1f}%")
        if face_result.document_face_ratio < 0.01:
            st.warning("Rostro del DNI muy pequeño. Acerca el documento o usa mejor luz.")
        st.json(face_result.to_dict())
        st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("Texto OCR crudo"):
        st.text(ocr_result.full_text)

    with st.expander("JSON completo"):
        st.code(json.dumps(result_payload, indent=2, ensure_ascii=False), language="json")


def main() -> None:
    settings = Settings.from_env()
    settings.configure_runtime()

    st.set_page_config(
        page_title="GLADiiUM Identity Verification",
        page_icon=str(LOGO_PATH) if LOGO_PATH.exists() else None,
        layout="wide",
    )

    st.markdown(build_css(False), unsafe_allow_html=True)

    # Mobile route: short branded header, no sidebar params.
    qp = st.query_params
    if qp.get("mobile") == "1":
        sid = qp.get("sid") or ""
        cleanup_old()
        st.markdown(
            f"""
            <div class="gld-brand">
              <div>
                <div class="gld-logo-wrap" style="display:inline-flex;">
                  {('<img src="' + _logo_data_uri() + '" />') if _logo_data_uri() else 'GLADIIUM'}
                </div>
              </div>
              <div class="gld-pill">Mobile</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        render_mobile_capture(sid)
        return

    cleanup_old()
    render_brand_header()

    with st.sidebar:
        st.header("Parametros")
        face_match_threshold = st.slider(
            "Umbral match facial",
            min_value=0.10,
            max_value=0.90,
            value=float(settings.face_match_threshold),
            step=0.01,
            help="Por encima de este valor: match. Recomendado 0.36–0.42 para DNI vs selfie.",
        )
        face_review_threshold = st.slider(
            "Umbral revisión facial",
            min_value=0.10,
            max_value=0.90,
            value=float(settings.face_review_threshold),
            step=0.01,
            help="Entre revisión y match: requiere revisión manual. Recomendado 0.22–0.28.",
        )
        st.caption(
            "Defaults sugeridos para DNI vs selfie: match 0.38, revision 0.25. "
            "Cross-domain (foto impresa pequeña vs selfie HD) baja el rango típico."
        )

    st.markdown('<div class="gld-step">Paso <b>01</b> · Modo de captura</div>', unsafe_allow_html=True)
    capture_mode = st.radio(
        "Modo de captura",
        options=["Subir archivo"],
        horizontal=True,
        label_visibility="collapsed",
    )

    st.markdown('<div class="gld-step">Paso <b>02</b> · Documento y rostro</div>', unsafe_allow_html=True)

    if capture_mode == "Desde mi teléfono (QR)":
        qr_doc_bytes, qr_selfie_bytes = render_qr_handoff()
        if not (qr_doc_bytes and qr_selfie_bytes):
            return
        st.markdown('<div class="gld-step">Paso <b>03</b> · Verificación</div>', unsafe_allow_html=True)
        if not st.button("Iniciar verificación", type="primary", key="qr_run"):
            return
        _run_pipeline(
            settings=settings,
            doc_bytes=qr_doc_bytes,
            selfie_bytes=qr_selfie_bytes,
            face_match_threshold=face_match_threshold,
            face_review_threshold=face_review_threshold,
        )
        return

    col_doc, col_selfie = st.columns(2, gap="large")
    with col_doc:
        st.markdown('<div class="gld-card"><h3>DNI · frente</h3>', unsafe_allow_html=True)
        document_file = st.file_uploader(
            "Imagen frontal del DNI (JPG, PNG, HEIC, WEBP)",
            type=UPLOAD_TYPES,
            key="doc_upl",
        )
        if document_file:
            try:
                st.image(
                    _read_image_bytes(document_file),
                    caption="DNI cargado",
                    use_column_width=True,
                )
            except ValueError as exc:
                st.error(str(exc))
                document_file = None
        st.markdown(
            '<div class="gld-hint">Buena luz · sin reflejos · 4 esquinas visibles</div></div>',
            unsafe_allow_html=True,
        )

    with col_selfie:
        st.markdown('<div class="gld-card"><h3>Selfie</h3>', unsafe_allow_html=True)
        selfie_file = st.file_uploader(
            "Selfie o foto de la persona (JPG, PNG, HEIC, WEBP)",
            type=UPLOAD_TYPES,
            key="selfie_upl",
        )
        if selfie_file:
            try:
                st.image(
                    _read_image_bytes(selfie_file),
                    caption="Selfie cargada",
                    use_column_width=True,
                )
            except ValueError as exc:
                st.error(str(exc))
                selfie_file = None
        st.markdown(
            '<div class="gld-hint">Sin gorra ni lentes · rostro centrado y enfocado</div></div>',
            unsafe_allow_html=True,
        )

    if not document_file or not selfie_file:
        st.info("Sube una imagen del DNI y una selfie para iniciar la verificación.")
        return

    st.markdown('<div class="gld-step">Paso <b>03</b> · Verificación</div>', unsafe_allow_html=True)
    if not st.button("Iniciar verificación", type="primary"):
        return

    try:
        doc_bytes = _read_image_bytes(document_file)
        selfie_bytes = _read_image_bytes(selfie_file)
    except ValueError as exc:
        st.error(str(exc))
        return

    _run_pipeline(
        settings=settings,
        doc_bytes=doc_bytes,
        selfie_bytes=selfie_bytes,
        face_match_threshold=face_match_threshold,
        face_review_threshold=face_review_threshold,
    )


if __name__ == "__main__":
    main()
