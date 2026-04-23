def mock_lead_capture(name: str, email: str, platform: str) -> str:
    """Mock API function to capture a lead."""
    print(f"\n[SYSTEM LOG] Lead captured successfully: {name}, {email}, {platform}")
    return "Thank you! Our team will reach out to you shortly to get you started."