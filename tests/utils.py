import sys


def not_on_linux():
    """Return True if the current platform is not Linux.

    This is to avoid / alter test bahavior for non-Linux (as browser testing gets
    tricky on other platforms).
    """
    return not sys.platform.startswith("linux")
