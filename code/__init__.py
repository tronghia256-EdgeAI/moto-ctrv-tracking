"""CAAP project package.

NOTE: this package is named `code`, which shadows the stdlib `code` module
(used by pdb/InteractiveConsole). Scripts must run from the repo root, and
tests rely on tests/conftest.py purging a pre-imported stdlib `code` from
sys.modules. Avoid using pdb's interactive console inside this project.
"""
