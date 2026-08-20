"""Helpers for safely passing terminal text to UTF-8 JSON APIs."""


def normalize_terminal_text(text: str) -> str:
    """Repair UTF-8 bytes decoded by Python with ``surrogateescape``.

    Python may represent undecodable terminal bytes as low-surrogate characters.
    HTTP clients correctly refuse to serialize those characters as JSON.  Valid
    UTF-8 input is returned unchanged; malformed input is repaired with the
    Unicode replacement character rather than crashing a workflow.
    """
    if not any(0xD800 <= ord(char) <= 0xDFFF for char in text):
        return text

    try:
        raw_bytes = text.encode("utf-8", errors="surrogateescape")
    except UnicodeEncodeError:
        return text.encode("utf-8", errors="replace").decode("utf-8")
    return raw_bytes.decode("utf-8", errors="replace")
