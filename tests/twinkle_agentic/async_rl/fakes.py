from collections import defaultdict
from typing import Any, Dict


class FakeTransferQueueClient:
    """Test-only object that mimics the TransferQueue KV API."""

    def __init__(self):
        self.fields: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
        self.tags: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)

    def kv_put(self, key: str, partition_id: str, fields=None, tag=None):
        if fields:
            current = dict(self.fields[partition_id].get(key) or {})
            current.update(dict(fields))
            self.fields[partition_id][key] = current
        elif key not in self.fields[partition_id]:
            self.fields[partition_id][key] = {}
        if tag:
            current_tag = dict(self.tags[partition_id].get(key) or {})
            current_tag.update(dict(tag))
            self.tags[partition_id][key] = current_tag

    def kv_batch_put(self, keys, partition_id: str, fields=None, tags=None):
        if isinstance(keys, str):
            keys = [keys]
        for i, key in enumerate(keys):
            f = None
            if fields is not None:
                if isinstance(fields, dict):
                    f = {k: v[i] if isinstance(v, list) else v for k, v in fields.items()}
                elif hasattr(fields, 'batch_size'):
                    f = {k: fields[k][i] for k in fields.keys()}
            t = tags[i] if tags is not None else None
            self.kv_put(key, partition_id, fields=f, tag=t)

    def kv_batch_get(self, keys, partition_id: str, select_fields=None):
        if isinstance(keys, str):
            keys = [keys]
        selected_fields = select_fields
        if isinstance(selected_fields, str):
            selected_fields = [selected_fields]
        rows = [dict(self.fields[partition_id].get(key) or {}) for key in keys]
        field_names = set()
        for row in rows:
            field_names.update(row)
        if selected_fields is not None:
            field_names.intersection_update(selected_fields)
        return {field_name: [row.get(field_name) for row in rows] for field_name in field_names}

    def kv_list(self, partition_id=None):
        if partition_id is not None:
            return {partition_id: dict(self.tags.get(partition_id) or {})}
        return {pid: dict(tags) for pid, tags in self.tags.items()}

    def kv_clear(self, keys, partition_id: str):
        if isinstance(keys, str):
            keys = [keys]
        for key in keys:
            self.fields.get(partition_id, {}).pop(key, None)
            self.tags.get(partition_id, {}).pop(key, None)
