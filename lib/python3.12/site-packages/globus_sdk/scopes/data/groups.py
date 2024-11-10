from ..builder import ScopeBuilder

GroupsScopes = ScopeBuilder(
    "groups.api.globus.org",
    known_scopes=[
        "all",
        "view_my_groups_and_memberships",
    ],
)
"""Groups scopes.

.. listknownscopes:: globus_sdk.scopes.GroupsScopes
"""


NexusScopes = ScopeBuilder(
    "nexus.api.globus.org",
    known_scopes=[
        "groups",
    ],
)
"""Nexus scopes (internal use only).

.. listknownscopes:: globus_sdk.scopes.NexusScopes
"""
