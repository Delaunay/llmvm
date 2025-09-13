import llmvm.plugins
from llmvm.core import discover_plugins


def test_plugins():
    plugins = discover_plugins(llmvm.plugins)

    assert len(plugins) == 1
