from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher


DEPARTMENTS: tuple[str, ...] = (
    "ATLANTIDA",
    "COLON",
    "COMAYAGUA",
    "COPAN",
    "CORTES",
    "CHOLUTECA",
    "EL PARAISO",
    "FRANCISCO MORAZAN",
    "GRACIAS A DIOS",
    "INTIBUCA",
    "ISLAS DE LA BAHIA",
    "LA PAZ",
    "LEMPIRA",
    "OCOTEPEQUE",
    "OLANCHO",
    "SANTA BARBARA",
    "VALLE",
    "YORO",
)


MUNICIPALITY_TO_DEPARTMENT: dict[str, str] = {
    # Francisco Morazan
    "TEGUCIGALPA": "FRANCISCO MORAZAN",
    "COMAYAGUELA": "FRANCISCO MORAZAN",
    "DISTRITO CENTRAL": "FRANCISCO MORAZAN",
    "TALANGA": "FRANCISCO MORAZAN",
    "VALLE DE ANGELES": "FRANCISCO MORAZAN",
    # Cortes
    "SAN PEDRO SULA": "CORTES",
    "CHOLOMA": "CORTES",
    "VILLANUEVA": "CORTES",
    "LA LIMA": "CORTES",
    "PUERTO CORTES": "CORTES",
    "OMOA": "CORTES",
    "POTRERILLOS": "CORTES",
    "PIMIENTA": "CORTES",
    # Atlantida
    "LA CEIBA": "ATLANTIDA",
    "TELA": "ATLANTIDA",
    "JUTIAPA": "ATLANTIDA",
    "ESPARTA": "ATLANTIDA",
    # Yoro
    "EL PROGRESO": "YORO",
    "YORO": "YORO",
    "OLANCHITO": "YORO",
    "MOROCELI": "YORO",
    # Comayagua
    "SIGUATEPEQUE": "COMAYAGUA",
    "COMAYAGUA": "COMAYAGUA",
    "LA LIBERTAD": "COMAYAGUA",
    # El Paraiso
    "DANLI": "EL PARAISO",
    "YUSCARAN": "EL PARAISO",
    "EL PARAISO": "EL PARAISO",
    # Olancho
    "JUTICALPA": "OLANCHO",
    "CATACAMAS": "OLANCHO",
    "CAMPAMENTO": "OLANCHO",
    "MANGULILE": "OLANCHO",
    # Islas de la Bahia
    "ROATAN": "ISLAS DE LA BAHIA",
    "UTILA": "ISLAS DE LA BAHIA",
    "GUANAJA": "ISLAS DE LA BAHIA",
    # Copan
    "SANTA ROSA DE COPAN": "COPAN",
    "COPAN RUINAS": "COPAN",
    "LA UNION": "COPAN",
    # Lempira
    "GRACIAS": "LEMPIRA",
    "ERANDIQUE": "LEMPIRA",
    # Ocotepeque
    "OCOTEPEQUE": "OCOTEPEQUE",
    "NUEVA OCOTEPEQUE": "OCOTEPEQUE",
    # Valle
    "NACAOME": "VALLE",
    "AMAPALA": "VALLE",
    # La Paz
    "LA PAZ": "LA PAZ",
    "MARCALA": "LA PAZ",
    # Gracias a Dios
    "PUERTO LEMPIRA": "GRACIAS A DIOS",
    "BRUS LAGUNA": "GRACIAS A DIOS",
    # Colon
    "TRUJILLO": "COLON",
    "TOCOA": "COLON",
    "SONAGUERA": "COLON",
    # Choluteca
    "CHOLUTECA": "CHOLUTECA",
    "MARCOVIA": "CHOLUTECA",
    # Intibuca
    "LA ESPERANZA": "INTIBUCA",
    "INTIBUCA": "INTIBUCA",
    # Santa Barbara
    "SANTA BARBARA": "SANTA BARBARA",
    "TRINIDAD": "SANTA BARBARA",
}


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^A-Za-z0-9 ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip().upper()
    return text


def _best_window_ratio(canonical: str, tokens: list[str]) -> float:
    n = len(canonical.split())
    if n == 0 or len(tokens) < n:
        return 0.0
    best = 0.0
    for i in range(len(tokens) - n + 1):
        window = " ".join(tokens[i : i + n])
        ratio = SequenceMatcher(None, canonical, window).ratio()
        if ratio > best:
            best = ratio
    return best


def match_place(text: str, cutoff: float = 0.78) -> dict[str, str | None]:
    """Match free-form OCR text to canonical Honduras places.

    Returns a dict with keys: municipality, department, country.
    Each value is the canonical name when fuzzy ratio >= cutoff, else None.
    """
    empty = {"municipality": None, "department": None, "country": None}
    if not text:
        return empty

    norm = _normalize(text)
    if not norm:
        return empty

    tokens = norm.split()
    country = "HONDURAS" if "HONDURAS" in norm else None
    if country:
        tokens = [t for t in tokens if t != "HONDURAS"]

    best_muni: tuple[float, str] | None = None
    for muni in MUNICIPALITY_TO_DEPARTMENT:
        ratio = _best_window_ratio(muni, tokens)
        if ratio < cutoff:
            continue
        if (
            best_muni is None
            or ratio > best_muni[0]
            or (ratio == best_muni[0] and len(muni) > len(best_muni[1]))
        ):
            best_muni = (ratio, muni)

    best_dept: tuple[float, str] | None = None
    for dept in DEPARTMENTS:
        ratio = _best_window_ratio(dept, tokens)
        if ratio < cutoff:
            continue
        if (
            best_dept is None
            or ratio > best_dept[0]
            or (ratio == best_dept[0] and len(dept) > len(best_dept[1]))
        ):
            best_dept = (ratio, dept)

    municipality: str | None = None
    department: str | None = None
    if best_muni and best_dept:
        muni_dept = MUNICIPALITY_TO_DEPARTMENT[best_muni[1]]
        if muni_dept == best_dept[1]:
            municipality = best_muni[1]
            department = best_dept[1]
        elif best_muni[0] > best_dept[0] or (
            best_muni[0] == best_dept[0] and len(best_muni[1]) > len(best_dept[1])
        ):
            municipality = best_muni[1]
            department = MUNICIPALITY_TO_DEPARTMENT[best_muni[1]]
        else:
            department = best_dept[1]
    elif best_muni:
        municipality = best_muni[1]
        department = MUNICIPALITY_TO_DEPARTMENT[best_muni[1]]
    elif best_dept:
        department = best_dept[1]

    return {
        "municipality": municipality,
        "department": department,
        "country": country,
    }
