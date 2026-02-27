import pytest


def pytest_addoption(parser):
    parser.addoption("--pg-host", default="localhost", help="PostgreSQL host for Apache AGE tests")
    parser.addoption("--pg-port", default="5432", help="PostgreSQL port for Apache AGE tests")
    parser.addoption("--pg-user", default="postgres", help="PostgreSQL user for Apache AGE tests")
    parser.addoption("--pg-password", default="postgres", help="PostgreSQL password for Apache AGE tests")
    parser.addoption("--pg-database", default="postgres", help="PostgreSQL database for Apache AGE tests")


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Store test outcome on the item so fixtures can read it."""
    outcome = yield
    rep = outcome.get_result()
    setattr(item, f"rep_{rep.when}", rep)
