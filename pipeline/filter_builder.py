"""
DashVector filter builder.

Builds DashVector server-side filter strings for scalar fields.
Array field filters (objects, actions) are handled client-side
 (contain_any / contain_all not supported on Singapore cluster).

Supported server-side operations:
  - Equality:    field = value,  field != value
  - Comparison:  field >/>=/</<= value
  - Boolean:     field = true/false
  - String like: field like "prefix%"
  - Logical:     and, or
"""

from dataclasses import dataclass, field
from typing import List, Optional, Union


class FilterBuilder:
    """Builds DashVector SQL-like filter strings for scalar fields."""

    def __init__(self):
        self._conditions: List[str] = []

    def _add(self, condition: str) -> "FilterBuilder":
        self._conditions.append(condition)
        return self

    def eq(self, field: str, value: Union[str, int, float, bool]) -> "FilterBuilder":
        if isinstance(value, str):
            return self._add(f'{field} = "{value}"')
        elif isinstance(value, bool):
            return self._add(f"{field} = {str(value).lower()}")
        else:
            return self._add(f"{field} = {value}")

    def ne(self, field: str, value: Union[str, int, float, bool]) -> "FilterBuilder":
        if isinstance(value, str):
            return self._add(f'{field} != "{value}"')
        elif isinstance(value, bool):
            return self._add(f"{field} != {str(value).lower()}")
        else:
            return self._add(f"{field} != {value}")

    def gt(self, field: str, value: Union[int, float]) -> "FilterBuilder":
        return self._add(f"{field} > {value}")

    def gte(self, field: str, value: Union[int, float]) -> "FilterBuilder":
        return self._add(f"{field} >= {value}")

    def lt(self, field: str, value: Union[int, float]) -> "FilterBuilder":
        return self._add(f"{field} < {value}")

    def lte(self, field: str, value: Union[int, float]) -> "FilterBuilder":
        return self._add(f"{field} <= {value}")

    def like(self, field: str, pattern: str) -> "FilterBuilder":
        return self._add(f'{field} like "{pattern}"')

    def build(self) -> str:
        if not self._conditions:
            return ""
        return " AND ".join(self._conditions)


@dataclass
class FilterParams:
    """
    Combined filter parameters for hybrid retrieval.

    server_filter: DashVector-side SQL-like filter (scalar fields only).
    objects: Client-side objects filter (applied in Python).
    actions: Client-side actions filter (applied in Python).
    require_all_objects: If True, require ALL objects.
    require_all_actions: If True, require ALL actions.
    """
    server_filter: str = ""
    objects: List[str] = field(default_factory=list)
    actions: List[str] = field(default_factory=list)
    require_all_objects: bool = False
    require_all_actions: bool = False

    @property
    def has_client_filter(self) -> bool:
        return bool(self.objects or self.actions)


def build_filter(
    objects: Optional[List[str]] = None,
    actions: Optional[List[str]] = None,
    lighting: Optional[str] = None,
    occlusion: Optional[str] = None,
    is_anomaly: Optional[bool] = None,
    category: Optional[str] = None,
    video_id: Optional[str] = None,
    episode_id: Optional[str] = None,
    view_type: Optional[str] = None,
    require_all_objects: bool = False,
    require_all_actions: bool = False,
) -> FilterParams:
    """
    Build combined filter params (server-side + client-side).

    Scalar filters (lighting, occlusion, etc.) go to DashVector.
    Array filters (objects, actions) are applied client-side in Python.

    Returns:
        FilterParams with server_filter and client-side filter arrays.
    """
    fb = FilterBuilder()

    if lighting:
        fb.eq("lighting", lighting)
    if occlusion:
        fb.eq("occlusion", occlusion)
    if is_anomaly is not None:
        fb.eq("is_anomaly", is_anomaly)
    if category:
        fb.eq("category", category)
    if video_id:
        fb.eq("video_id", video_id)
    if episode_id:
        fb.eq("episode_id", episode_id)
    if view_type:
        fb.eq("view_type", view_type)

    return FilterParams(
        server_filter=fb.build(),
        objects=objects or [],
        actions=actions or [],
        require_all_objects=require_all_objects,
        require_all_actions=require_all_actions,
    )
