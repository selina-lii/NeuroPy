import importlib

_LAZY = {'CCGReviewUI': ('neuropy.ui.ccg_ui', 'CCGReviewUI'),
         'PromptSpace': ('neuropy.productivity.prompt_space', 'PromptSpace')}


def __getattr__(name):
    """Lazy: importing a submodule must not drag the whole GUI (and a Qt backend) in."""
    if name not in _LAZY:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, attr = _LAZY[name]
    return getattr(importlib.import_module(module), attr)
