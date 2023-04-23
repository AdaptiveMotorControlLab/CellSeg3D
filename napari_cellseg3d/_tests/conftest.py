import os

import pytest


@pytest.fixture(scope="session", autouse=True)
def env_config():
    """
    Configure environment variables needed for the test session
    """

    # This makes QT render everything offscreen and thus prevents
    # any Modals / Dialogs or other Widgets being rendered on the screen while running unit tests
    os.environ["QT_QPA_PLATFORM"] = "offscreen"

    yield

    os.environ.pop("QT_QPA_PLATFORM")
