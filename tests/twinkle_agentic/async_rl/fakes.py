from collections import defaultdict
from typing import Any, Dict


class FakeTransferQueueClient:
    """Test-only object that mimics the TransferQueue KV API."""

    def __init__(self):
        self.fields: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
        self.tags: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
        self.kv_batch_put_calls: list[dict[str, Any]] = []

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
        self.kv_batch_put_calls.append({
            'keys': list(keys),
            'partition_id': partition_id,
            'has_fields': fields is not None,
            'has_tags': tags is not None,
        })
        fields = self._rows_from_fields(fields, len(keys))
        tags = tags or [{} for _ in keys]
        for key, row_fields, tag in zip(keys, fields, tags):
            self.kv_put(key=key, partition_id=partition_id, fields=row_fields, tag=tag)

    async def async_kv_batch_put(self, keys, partition_id: str, fields=None, tags=None, data_parser=None):
        self.kv_batch_put(keys=keys, partition_id=partition_id, fields=fields, tags=tags)

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

    @staticmethod
    def _rows_from_fields(fields, size: int):
        if fields is None:
            return [{} for _ in range(size)]
        if hasattr(fields, 'to_dict'):
            fields = fields.to_dict()
        if isinstance(fields, list):
            return [dict(item) for item in fields]
        if isinstance(fields, dict):
            rows = [dict() for _ in range(size)]
            for field_name, value in fields.items():
                values = FakeTransferQueueClient._field_values(value, size)
                for row, item in zip(rows, values):
                    row[field_name] = item
            return rows
        return [{'data': fields} for _ in range(size)]

    @staticmethod
    def _field_values(value, size: int):
        if hasattr(value, 'unbind'):
            return list(value.unbind(0))
        if hasattr(value, 'tolist'):
            value = value.tolist()
        if isinstance(value, list) and len(value) == size:
            return value
        try:
            values = list(value)
            if len(values) == size:
                return values
        except TypeError:
            pass
        return [value for _ in range(size)]
