from network_cpt.clean import exact_fingerprint, normalize_text, quality_rejection_reason
from network_cpt.schema import Document


def make_document(text: str, license_name: str = "MIT") -> Document:
    return Document(
        id="doc-1",
        source="test",
        document="bgp-guide",
        title="BGP",
        license=license_name,
        language="en",
        text=text,
        snapshot="abc123",
    )


def test_normalize_removes_navigation_but_keeps_indentation() -> None:
    text = "Table of Contents\n\nConfig:\n   router bgp 65000\n\n\n\nNext"
    cleaned = normalize_text(text)

    assert "Table of Contents" not in cleaned
    assert "Next" not in cleaned
    assert "   router bgp 65000" in cleaned


def test_exact_fingerprint_ignores_case_and_whitespace() -> None:
    assert exact_fingerprint("BGP  Route\n") == exact_fingerprint("bgp route")


def test_quality_rejects_missing_license() -> None:
    text = "BGP route selection and path attributes. " * 10
    assert quality_rejection_reason(make_document(text, "")) == "missing_license"
