from ..builder import ScopeBuilder

TransferScopes = ScopeBuilder(
    "transfer.api.globus.org",
    known_scopes=[
        "all",
        "gcp_install",
    ],
)
"""Globus Transfer scopes.

.. listknownscopes:: globus_sdk.scopes.TransferScopes
"""
