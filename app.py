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
from src.image_io import image_bytes_to_bgr
from src.ocr import OCRError, run_ocr
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
    """Return JPEG-normalized bytes from an uploaded file or camera capture.

    HEIC/HEIF (iPhone) uploads are decoded via Pillow + pillow-heif and
    re-encoded as JPEG so downstream OCR, face matcher and st.image all
    consume the same format.
    """
    raw = uploaded.getvalue()
    name = (getattr(uploaded, "name", "") or "").lower()
    if name.endswith((".heic", ".heif")):
        import cv2

        bgr = image_bytes_to_bgr(raw)
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not ok:
            raise ValueError("No se pudo convertir HEIC a JPEG.")
        return buf.tobytes()
    return raw


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
  padding-top: 1.2rem;
  max-width: 1180px;
}}

/* Brand bar */
.gld-brand {{
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 6px 4px 18px 4px;
  border-bottom: 1px solid var(--border);
  margin-bottom: 28px;
  gap: 16px;
}}
.gld-logo-wrap {{
  background: var(--logo-bg);
  padding: {"8px 14px" if dark else "0"};
  border-radius: 10px;
  display: inline-flex;
  align-items: center;
}}
.gld-logo-wrap img {{
  height: 48px;
  width: auto;
  display: block;
}}
.gld-pill {{
  font-size: 11px;
  letter-spacing: 0.22em;
  text-transform: uppercase;
  color: var(--accent);
  border: 1px solid var(--border-strong);
  padding: 6px 12px;
  border-radius: 999px;
  background: var(--accent-soft);
  font-weight: 600;
}}

/* Hero */
.gld-hero h1 {{
  font-family: 'Poppins', sans-serif !important;
  font-weight: 700;
  letter-spacing: -0.01em;
  font-size: 38px;
  line-height: 1.1;
  margin: 8px 0 6px 0;
  color: var(--text);
}}
.gld-hero h1 em {{
  font-style: normal;
  color: var(--accent);
}}
.gld-hero p {{
  color: var(--muted);
  margin: 0;
  font-size: 15px;
  font-weight: 400;
}}

/* Section labels */
.gld-step {{
  display: inline-flex;
  align-items: center;
  gap: 10px;
  padding: 6px 12px;
  border: 1px solid var(--border);
  border-radius: 999px;
  font-size: 11px;
  letter-spacing: 0.2em;
  text-transform: uppercase;
  color: var(--text);
  background: var(--surface);
  margin: 26px 0 12px 0;
  font-weight: 600;
}}
.gld-step b {{
  color: var(--accent);
  font-weight: 700;
}}

/* Capture card */
.gld-card {{
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 18px;
  box-shadow: var(--shadow);
}}
.gld-card h3 {{
  font-family: 'Poppins', sans-serif;
  font-weight: 600;
  letter-spacing: 0;
  color: var(--text);
  margin: 0 0 12px 0;
  font-size: 16px;
}}
.gld-card .gld-hint {{
  color: var(--muted);
  font-size: 12px;
  margin-top: 12px;
  border-top: 1px dashed var(--border);
  padding-top: 10px;
  font-weight: 400;
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
  border-radius: 10px;
  font-weight: 600;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  padding: 12px 26px;
  box-shadow: 0 6px 18px rgba(122,20,20,0.25);
  transition: transform .08s ease, box-shadow .2s ease, filter .15s ease;
  font-family: 'Poppins', sans-serif !important;
}}
.stButton > button:hover {{
  transform: translateY(-1px);
  filter: brightness(1.06);
  box-shadow: 0 10px 26px rgba(122,20,20,0.35);
}}

/* Radio */
[role="radiogroup"] label {{
  border: 1px solid var(--border);
  border-radius: 999px;
  padding: 6px 14px;
  margin-right: 8px;
  background: var(--surface);
  color: var(--text) !important;
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
  border-radius: 14px;
  padding: 18px 22px;
  margin: 6px 0 18px 0;
  border: 1px solid var(--border);
  display: flex; align-items: center; gap: 14px;
  font-weight: 600; letter-spacing: 0.14em;
  text-transform: uppercase;
  background: var(--surface);
}}
.gld-banner .dot {{
  width: 10px; height: 10px; border-radius: 50%;
  box-shadow: 0 0 12px currentColor;
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
          <h1>Verificacion de <em>Identidad</em></h1>
          <p>Captura en vivo del documento y del rostro. OCR documental y comparacion biometrica en un solo flujo.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_mobile_capture(sid: str) -> None:
    state = read_state(sid)
    if state is None:
        st.error("Sesion invalida o expirada. Pide al desktop que genere un QR nuevo.")
        return

    st.markdown(
        '<div class="gld-step">Captura desde tu telefono</div>',
        unsafe_allow_html=True,
    )

    doc_ready = state.get("document") == "ready"
    selfie_ready = state.get("selfie") == "ready"

    progress = []
    progress.append("DNI " + ("OK" if doc_ready else "pendiente"))
    progress.append("Selfie " + ("OK" if selfie_ready else "pendiente"))
    st.caption(" · ".join(progress))

    if not doc_ready:
        st.markdown('<div class="gld-card"><h3>Paso 1 de 2 · DNI (frente)</h3>', unsafe_allow_html=True)
        st.caption(
            "Buena luz, sin reflejos, las 4 esquinas visibles. Apoya el documento sobre superficie oscura."
        )
        photo = st.camera_input("Toma una foto del DNI", key="m_doc_cam")
        upload = st.file_uploader(
            "O sube desde galeria",
            type=UPLOAD_TYPES,
            key="m_doc_upl",
        )
        chosen = photo or upload
        if chosen is not None:
            try:
                normalized = _read_image_bytes(chosen)
            except ValueError as exc:
                st.error(str(exc))
                return
            save_image(sid, "document", normalized)
            st.success("DNI recibido. Sigue con la selfie.")
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
        return

    if not selfie_ready:
        st.markdown('<div class="gld-card"><h3>Paso 2 de 2 · Selfie</h3>', unsafe_allow_html=True)
        st.caption("Mira a camara con rostro centrado. Sin gorra, sin lentes oscuros, buena luz frontal.")
        photo = st.camera_input("Toma una selfie", key="m_self_cam")
        upload = st.file_uploader(
            "O sube desde galeria",
            type=UPLOAD_TYPES,
            key="m_self_upl",
        )
        chosen = photo or upload
        if chosen is not None:
            try:
                normalized = _read_image_bytes(chosen)
            except ValueError as exc:
                st.error(str(exc))
                return
            save_image(sid, "selfie", normalized)
            st.success("Selfie recibida.")
            st.rerun()
        if st.button("Volver a tomar el DNI", key="m_redo_doc"):
            reset_kind(sid, "document")
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
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
        st.markdown('<div class="gld-card"><h3>Captura desde tu telefono</h3>', unsafe_allow_html=True)
        st.markdown(
            "1. Escanea el QR con la camara del telefono.<br>"
            "2. Toma la foto del DNI con buena luz.<br>"
            "3. Toma la selfie centrando el rostro.<br>"
            "4. Esta pagina se actualiza sola.",
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
                st.image(doc_bytes, use_container_width=True)
        else:
            st.caption("Pendiente — captura desde el telefono.")
        st.markdown("</div>", unsafe_allow_html=True)
    with status_col2:
        st.markdown('<div class="gld-card"><h3>Selfie</h3>', unsafe_allow_html=True)
        if selfie_ready:
            selfie_bytes = read_image(sid, "selfie")
            if selfie_bytes:
                st.image(selfie_bytes, use_container_width=True)
        else:
            st.caption("Pendiente — captura desde el telefono.")
        st.markdown("</div>", unsafe_allow_html=True)

    if not (doc_ready and selfie_ready):
        try:
            from streamlit_autorefresh import st_autorefresh

            st_autorefresh(interval=2000, key=f"qr_poll_{sid}")
        except ImportError:
            st.caption("streamlit-autorefresh no esta instalado; refresca la pagina manualmente.")
        return None, None

    return read_image(sid, "document"), read_image(sid, "selfie")


def render_decision(decision: str) -> None:
    mapping = {
        "verified": ("gld-verified", "Identidad verificada"),
        "manual_review": ("gld-review", "Requiere revision manual"),
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


def _run_pipeline(
    settings: Settings,
    doc_bytes: bytes,
    selfie_bytes: bytes,
    face_match_threshold: float,
    face_review_threshold: float,
) -> None:
    with st.spinner("Ejecutando OCR..."):
        try:
            ocr_result = run_ocr(doc_bytes, lang=settings.ocr_lang)
        except OCRError as exc:
            st.error(str(exc))
            return

    parsed = parse_honduras_dni(ocr_result.full_text, ocr_result.average_confidence)

    with st.spinner("Comparando rostros..."):
        try:
            matcher = FaceMatcher(
                model_name=settings.face_model_name,
                match_threshold=face_match_threshold,
                review_threshold=face_review_threshold,
                model_root=settings.insightface_root,
            )
            face_result = matcher.compare(
                image_bytes_to_bgr(doc_bytes),
                image_bytes_to_bgr(selfie_bytes),
            )
        except (FaceMatcherError, ValueError) as exc:
            st.error(str(exc))
            return

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
        st.markdown('<div class="gld-card"><h3>Resultado biometrico</h3>', unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        m1.metric("Similitud", f"{face_result.similarity:.3f}")
        m2.metric("Rostro DNI", f"{face_result.document_face_ratio*100:.1f}%")
        m3.metric("Rostro selfie", f"{face_result.selfie_face_ratio*100:.1f}%")
        if face_result.document_face_ratio < 0.01:
            st.warning("Rostro del DNI muy pequeno. Acerca el documento o usa mejor luz.")
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
        page_title="Gladiium · Identity Verification",
        page_icon=str(LOGO_PATH) if LOGO_PATH.exists() else None,
        layout="wide",
    )

    if "dark_mode" not in st.session_state:
        st.session_state["dark_mode"] = False

    # Theme toggle (top right)
    _, theme_col = st.columns([8, 1])
    with theme_col:
        st.session_state["dark_mode"] = st.toggle(
            "Oscuro",
            value=st.session_state["dark_mode"],
            key="dark_toggle",
        )

    st.markdown(build_css(st.session_state["dark_mode"]), unsafe_allow_html=True)

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
            help="Por encima de este valor: match. Recomendado 0.36-0.42 para DNI vs selfie.",
        )
        face_review_threshold = st.slider(
            "Umbral revision facial",
            min_value=0.10,
            max_value=0.90,
            value=float(settings.face_review_threshold),
            step=0.01,
            help="Entre revision y match: requiere revision manual. Recomendado 0.22-0.28.",
        )
        st.caption(
            "Defaults sugeridos para DNI vs selfie: match 0.38, revision 0.25. "
            "Cross-domain (foto impresa pequena vs selfie HD) baja el rango tipico."
        )

    st.markdown('<div class="gld-step">Paso <b>01</b> · Modo de captura</div>', unsafe_allow_html=True)
    capture_mode = st.radio(
        "Modo de captura",
        options=["Camara en vivo", "Desde mi telefono (QR)", "Subir archivo"],
        horizontal=True,
        label_visibility="collapsed",
    )

    st.markdown('<div class="gld-step">Paso <b>02</b> · Documento y rostro</div>', unsafe_allow_html=True)

    if capture_mode == "Desde mi telefono (QR)":
        qr_doc_bytes, qr_selfie_bytes = render_qr_handoff()
        if not (qr_doc_bytes and qr_selfie_bytes):
            return
        st.markdown('<div class="gld-step">Paso <b>03</b> · Verificacion</div>', unsafe_allow_html=True)
        if not st.button("Iniciar verificacion", type="primary", key="qr_run"):
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
        st.markdown('<div class="gld-card"><h3>Documento · DNI (frente)</h3>', unsafe_allow_html=True)
        if capture_mode == "Camara en vivo":
            document_file = st.camera_input(
                "Encuadra el DNI dentro del marco y captura",
                key="doc_cam",
            )
        else:
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
                        use_container_width=True,
                    )
                except ValueError as exc:
                    st.error(str(exc))
                    document_file = None
        st.markdown(
            '<div class="gld-hint">Buena luz. Sin reflejos. Las cuatro esquinas visibles.</div></div>',
            unsafe_allow_html=True,
        )

    with col_selfie:
        st.markdown('<div class="gld-card"><h3>Persona · Selfie</h3>', unsafe_allow_html=True)
        if capture_mode == "Camara en vivo":
            selfie_file = st.camera_input(
                "Mira a camara con rostro centrado",
                key="selfie_cam",
            )
        else:
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
                        use_container_width=True,
                    )
                except ValueError as exc:
                    st.error(str(exc))
                    selfie_file = None
        st.markdown(
            '<div class="gld-hint">Sin gorra ni lentes oscuros. Rostro completo y enfocado.</div></div>',
            unsafe_allow_html=True,
        )

    if not document_file or not selfie_file:
        if capture_mode == "Camara en vivo":
            st.info("Captura el DNI y la selfie con la camara para iniciar la verificacion.")
        else:
            st.info("Sube una imagen del DNI y una selfie para iniciar la verificacion.")
        return

    st.markdown('<div class="gld-step">Paso <b>03</b> · Verificacion</div>', unsafe_allow_html=True)
    if not st.button("Iniciar verificacion", type="primary"):
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
