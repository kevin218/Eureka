# conftest.py
def pytest_collection_modifyitems(session, config, items):
    """Modifies test items to ensure test functions run in a given order

    Parameters
    ----------
    session : pytest.Session
        The pytest session object.
    config : pytest.Config
        The pytest config object.
    items : List[pytest.Item]
        List of item objects.
    """
    function_order = ["test_trim", "test_medstddev",
                      "test_parameter", "test_parameters", "test_model",
                      "test_compositemodel", "test_polynomialmodel",
                      "test_transitmodel", "test_eclipsemodel",
                      "test_sinsoidalmodel", "test_poettr_model",
                      "test_poetecl_model", "test_poetpc_model",
                      "test_lorentzian_model", "test_exponentialmodel",
                      "test_simulation",
                      "test_MIRI", "test_NIRCam", "test_NIRSpec",
                      "test_NIRCamPhotometry", "test_NIRCamPhotometry_hex",
                      "test_WFC3"]
    item_names = [item.name for item in items]
    function_mapping = {item.name: item for item in items
                        if item.name in function_order}
    extra_functions = [item for item in items
                       if item.name not in function_order]

    sorted_items = []
    for func_ in function_order:
        if func_ in item_names:
            sorted_items.append(function_mapping[func_])
    sorted_items.extend(extra_functions)

    items[:] = sorted_items
