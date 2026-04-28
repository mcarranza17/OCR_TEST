import unittest

from src.dni_parser import parse_honduras_dni


class HondurasDNIParserTest(unittest.TestCase):
    def test_parses_common_honduras_dni_fields(self):
        raw_text = """
        REPUBLICA DE HONDURAS
        REGISTRO NACIONAL DE LAS PERSONAS
        DOCUMENTO NACIONAL DE IDENTIFICACION
        NOMBRE FORENAME MARIA JOSE
        APELLIDO SURNAME LOPEZ MEJIA
        FECHA DE NACIMIENTO DATE OF BIRTH 17/03/1992
        NUMERO DE NUMBER 0801 1992 12345
        FECHA DE EXPIRACION DATE OF EXPIRY 17/03/2032
        LUGAR DE NACIMIENTO PLACE OF BIRTH DISTRITO CENTRAL
        """

        parsed = parse_honduras_dni(raw_text, ocr_confidence=0.94)

        self.assertEqual(parsed.document_number, "0801-1992-12345")
        self.assertEqual(parsed.names, "MARIA JOSE")
        self.assertEqual(parsed.surnames, "LOPEZ MEJIA")
        self.assertEqual(parsed.birth_date, "17/03/1992")
        self.assertEqual(parsed.expiry_date, "17/03/2032")
        self.assertEqual(parsed.birth_place, "DISTRITO CENTRAL")
        self.assertEqual(parsed.birth_year_from_id, 1992)
        self.assertEqual(parsed.warnings, [])
        self.assertTrue(parsed.is_valid_for_demo)

    def test_flags_missing_name_fields(self):
        raw_text = "DOCUMENTO NACIONAL DE IDENTIFICACION\n0801-1992-12345"

        parsed = parse_honduras_dni(raw_text)

        self.assertEqual(parsed.document_number, "0801-1992-12345")
        self.assertIn("No se pudieron extraer nombres.", parsed.warnings)
        self.assertIn("No se pudieron extraer apellidos.", parsed.warnings)
        self.assertFalse(parsed.is_valid_for_demo)

    def test_flags_birth_year_mismatch(self):
        raw_text = """
        NOMBRE FORENAME ANA
        APELLIDO SURNAME PEREZ
        FECHA DE NACIMIENTO DATE OF BIRTH 01/01/1999
        NUMERO DE NUMBER 0801-1992-12345
        """

        parsed = parse_honduras_dni(raw_text)

        self.assertIn(
            "La fecha de nacimiento no coincide con el año del número de identidad.",
            parsed.warnings,
        )


if __name__ == "__main__":
    unittest.main()
