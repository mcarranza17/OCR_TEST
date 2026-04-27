from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass
from datetime import datetime

from .honduras_places import match_place


ID_RE = re.compile(r"(?<!\d)(\d{4})[\s-]?(\d{4})[\s-]?(\d{5})(?!\d)")
DATE_RE = re.compile(r"(?<!\d)(\d{1,2})[./-](\d{1,2})[./-](\d{2,4})(?!\d)")


NAME_LABELS: tuple[str, ...] = ("NOMBRE", "FORENAME", "NOMBRES")
SURNAME_LABELS: tuple[str, ...] = ("APELLIDO", "SURNAME", "APELLIDOS")
BIRTH_DATE_LABELS: tuple[str, ...] = (
    "FECHA DE NACIMIENTO",
    "DATE OF BIRTH",
    "FECHA DE NAC",
    "FECHA DE NAG",
    "FEGHA DE NAC",
    "FEGHA DE NAG",
    "FEGHA DE",
    "NACIMIENTO",
)
EXPIRY_LABELS: tuple[str, ...] = (
    "FECHA DE EXPIRACION",
    "FECHA DE VENCIMIENTO",
    "FECHA DE EMISION",
    "DATE OF EXPIRY",
    "VALID UNTIL",
    "EXPIRACION",
    "VENCIMIENTO",
    "EXPIRY",
    "FECHA DEY",
    "FECHA DE",
)
PLACE_LABELS: tuple[str, ...] = (
    "LUGAR DE NACIMIENTO",
    "PLACE OF BIRTH",
    "PEACE OF BIRTH",
    "PLACE 0F BIRTH",
    "PEACE 0F BIRTH",
    "LUGAR",
)
HEADER_LABELS: tuple[str, ...] = (
    "RNP",
    "REPUBLICA",
    "REGISTRO",
    "NACIONAL DE LAS PERSONAS",
    "PERSONAS",
    "DOCUMENTO",
    "IDENTIFICACION",
    "IDENTITY",
    "DNI",
    "NACIONALIDAD",
    "NATIONALITY",
    "CIDNALIDAD",
    "NATIONAGEY",
    "HND",
)


def _dedup_by_length_desc(*groups: tuple[str, ...]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for group in groups:
        for label in group:
            if label not in seen:
                seen.add(label)
                out.append(label)
    return sorted(out, key=len, reverse=True)


# Labels removed from a value line. Includes short generics like "FECHA DE",
# so they must be applied AFTER longer variants.
LABELS_TO_STRIP = _dedup_by_length_desc(
    NAME_LABELS,
    SURNAME_LABELS,
    BIRTH_DATE_LABELS,
    EXPIRY_LABELS,
    PLACE_LABELS,
    HEADER_LABELS,
)

# Lines that contain any of these are considered "label-like" and act as a
# stop boundary when reading multi-line values.
BREAK_LABELS = LABELS_TO_STRIP


@dataclass(frozen=True)
class HondurasDNI:
    document_number: str | None
    names: str | None
    surnames: str | None
    birth_date: str | None
    expiry_date: str | None
    birth_place: str | None
    birth_municipality: str | None
    birth_department: str | None
    birth_country: str | None
    birth_year_from_id: int | None
    ocr_confidence: float | None
    raw_text: str
    warnings: list[str]

    @property
    def full_name(self) -> str | None:
        parts = [part for part in (self.names, self.surnames) if part]
        return " ".join(parts) if parts else None

    @property
    def is_valid_for_demo(self) -> bool:
        return bool(self.document_number and self.full_name and not self.warnings)

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["full_name"] = self.full_name
        payload["is_valid_for_demo"] = self.is_valid_for_demo
        return payload


def parse_honduras_dni(raw_text: str, ocr_confidence: float | None = None) -> HondurasDNI:
    lines = [_normalize(line) for line in raw_text.splitlines() if line.strip()]
    joined = "\n".join(lines)

    document_number = _extract_document_number(joined)
    birth_year = _birth_year_from_doc(document_number)

    names = _clean(_extract_after_label(lines, NAME_LABELS, max_followup=3))
    surnames = _clean(_extract_after_label(lines, SURNAME_LABELS, max_followup=3))
    birth_place = _clean(_extract_after_label(lines, PLACE_LABELS, max_followup=3))
    birth_date, expiry_date = _extract_dates(joined, birth_year)

    place_match = match_place(birth_place or "")

    warnings = _validate(document_number, birth_year, names, surnames, birth_date)

    return HondurasDNI(
        document_number=document_number,
        names=names,
        surnames=surnames,
        birth_date=birth_date,
        expiry_date=expiry_date,
        birth_place=birth_place,
        birth_municipality=place_match["municipality"],
        birth_department=place_match["department"],
        birth_country=place_match["country"],
        birth_year_from_id=birth_year,
        ocr_confidence=ocr_confidence,
        raw_text=raw_text,
        warnings=warnings,
    )


def _normalize(value: str) -> str:
    ascii_text = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    ascii_text = re.sub(r"\s+", " ", ascii_text)
    return ascii_text.strip().upper()


def _extract_document_number(text: str) -> str | None:
    match = ID_RE.search(text)
    if not match:
        return None
    return "-".join(match.groups())


def _birth_year_from_doc(document_number: str | None) -> int | None:
    if not document_number:
        return None
    parts = document_number.split("-")
    if len(parts) != 3:
        return None
    try:
        return int(parts[1])
    except ValueError:
        return None


def _is_label_line(line: str) -> bool:
    return any(label in line for label in BREAK_LABELS)


def _strip_labels(line: str) -> str:
    cleaned = line
    for word in LABELS_TO_STRIP:
        cleaned = cleaned.replace(word, " ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" /-.,")
    return cleaned


def _has_letters(value: str) -> bool:
    return bool(re.search(r"[A-Z]", value))


def _extract_after_label(
    lines: list[str], labels: tuple[str, ...], max_followup: int = 2
) -> str | None:
    label_set = sorted(labels, key=len, reverse=True)
    for index, line in enumerate(lines):
        if not any(label in line for label in label_set):
            continue

        inline = _strip_labels(line)
        if _has_letters(inline):
            return inline

        collected: list[str] = []
        for nxt in lines[index + 1 : index + 1 + max_followup]:
            if _is_label_line(nxt):
                break
            cleaned = _strip_labels(nxt)
            if not _has_letters(cleaned):
                continue
            collected.append(cleaned)
        if collected:
            return " ".join(collected)
    return None


def _clean(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = re.sub(r"[^A-Z0-9 .-]", " ", value)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" -./")
    return cleaned or None


def _extract_dates(text: str, birth_year: int | None) -> tuple[str | None, str | None]:
    formatted: list[tuple[str, int]] = []
    for match in DATE_RE.finditer(text):
        day_s, month_s, year_s = match.groups()
        if len(year_s) == 2:
            year_s = f"20{year_s}" if int(year_s) < 40 else f"19{year_s}"
        try:
            day, month, year = int(day_s), int(month_s), int(year_s)
        except ValueError:
            continue
        if not (1 <= day <= 31 and 1 <= month <= 12 and 1900 <= year <= 2100):
            continue
        formatted.append((f"{day:02d}/{month:02d}/{year:04d}", year))

    if not formatted:
        return None, None

    birth: str | None = None
    if birth_year is not None:
        for value, year in formatted:
            if year == birth_year:
                birth = value
                break
    if birth is None:
        birth = min(formatted, key=lambda t: t[1])[0]

    current_year = datetime.now().year
    future_candidates = [t for t in formatted if t[1] > current_year and t[0] != birth]
    if future_candidates:
        expiry: str | None = max(future_candidates, key=lambda t: t[1])[0]
    else:
        expiry = next((value for value, _ in formatted if value != birth), None)
    return birth, expiry


def _validate(
    document_number: str | None,
    birth_year: int | None,
    names: str | None,
    surnames: str | None,
    birth_date: str | None,
) -> list[str]:
    warnings: list[str] = []

    if not document_number:
        warnings.append("No se encontro numero de identidad hondureno de 13 digitos.")

    current_year = datetime.now().year
    if birth_year is not None and not (1900 <= birth_year <= current_year):
        warnings.append("El ano dentro del numero de identidad no parece valido.")

    if birth_date and birth_year and not birth_date.endswith(str(birth_year)):
        warnings.append("La fecha de nacimiento no coincide con el ano del numero de identidad.")

    if not names:
        warnings.append("No se pudieron extraer nombres.")

    if not surnames:
        warnings.append("No se pudieron extraer apellidos.")

    return warnings
