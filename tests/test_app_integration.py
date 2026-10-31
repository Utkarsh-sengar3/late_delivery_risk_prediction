"""Integration tests for Streamlit App and pages."""

import pytest
from streamlit.testing.v1 import AppTest


def test_app_dashboard_loads():
    """Verify default landing dashboard loads cleanly without errors."""
    at = AppTest.from_file("app.py", default_timeout=20)
    at.run()
    assert len(at.exception) == 0
    assert at.session_state["page"] == "Dashboard"


def test_app_prediction_flow():
    """Verify prediction page form submission and history recording."""
    at = AppTest.from_file("app.py", default_timeout=20)
    at.run()
    at.session_state["page"] = "Model Prediction"
    at.run()
    assert len(at.exception) == 0

    forms = at.get("form")
    assert len(forms) > 0
    submit_btn = forms[0].get("button")[0]
    submit_btn.click().run()

    assert len(at.exception) == 0
    assert len(at.session_state["history"]) >= 1
    last = at.session_state["history"][0]
    assert "Probability" in last
    assert "Decision" in last


def test_all_pages_navigation():
    """Verify that all workspace and model pages load without unhandled exceptions."""
    at = AppTest.from_file("app.py", default_timeout=20)
    at.run()
    pages = [
        "Dashboard",
        "Model Prediction",
        "Prediction Analytics",
        "Model Performance",
        "Risk Analysis",
        "Explainability",
        "Prediction History",
        "About Website",
        "About Developer",
        "Settings",
    ]
    for p in pages:
        at.session_state["page"] = p
        at.run()
        assert len(at.exception) == 0, f"Exception found on page {p}"
