def test_can_import():
    import etf_kan
    from etf_kan.models import KANModel
    assert KANModel is not None
