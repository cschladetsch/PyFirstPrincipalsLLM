import json
import pickle
from pathlib import Path
from typing import Dict, Any, Union, Optional
from datetime import datetime
import threading

class ValueStorage:
    def __init__(self, storage_path: str = "math_values.db"):
        self.storage_path = Path(storage_path)
        self.values: Dict[str, Any] = {}
        self.metadata: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.Lock()
        self.load_from_disk()

    def set(self, name: str, value: Union[int, float, str], metadata: Optional[Dict[str, Any]] = None):
        with self.lock:
            self.values[name] = value
            self.metadata[name] = {
                'value': value,
                'type': type(value).__name__,
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat(),
                'access_count': 0,
                'custom_metadata': metadata or {}
            }
            self.save_to_disk()

    def get(self, name: str) -> Optional[Union[int, float, str]]:
        with self.lock:
            if name in self.values:
                self.metadata[name]['access_count'] += 1
                self.metadata[name]['last_accessed'] = datetime.now().isoformat()
                return self.values[name]
            return None

    def update(self, name: str, value: Union[int, float, str]):
        with self.lock:
            if name in self.values:
                old_value = self.values[name]
                self.values[name] = value
                self.metadata[name]['value'] = value
                self.metadata[name]['previous_value'] = old_value
                self.metadata[name]['updated_at'] = datetime.now().isoformat()
                self.save_to_disk()
                return True
            return False

    def delete(self, name: str) -> bool:
        with self.lock:
            if name in self.values:
                del self.values[name]
                del self.metadata[name]
                self.save_to_disk()
                return True
            return False

    def exists(self, name: str) -> bool:
        return name in self.values

    def list_all(self) -> Dict[str, Any]:
        with self.lock:
            return self.values.copy()

    def get_metadata(self, name: str) -> Optional[Dict[str, Any]]:
        with self.lock:
            return self.metadata.get(name)

    def clear_all(self):
        with self.lock:
            self.values.clear()
            self.metadata.clear()
            self.save_to_disk()

    def save_to_disk(self):
        data = {
            'values': self.values,
            'metadata': self.metadata
        }
        with open(self.storage_path, 'wb') as f:
            pickle.dump(data, f)

        json_path = self.storage_path.with_suffix('.json')
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)

    def load_from_disk(self):
        if self.storage_path.exists():
            try:
                with open(self.storage_path, 'rb') as f:
                    data = pickle.load(f)
                    self.values = data.get('values', {})
                    self.metadata = data.get('metadata', {})
            except Exception as e:
                print(f"Error loading storage: {e}")
                self.values = {}
                self.metadata = {}

    def search(self, pattern: str) -> Dict[str, Any]:
        with self.lock:
            results = {}
            for name, value in self.values.items():
                if pattern.lower() in name.lower():
                    results[name] = value
            return results

    def get_statistics(self) -> Dict[str, Any]:
        with self.lock:
            if not self.values:
                return {
                    'total_variables': 0,
                    'types': {},
                    'most_accessed': None
                }

            type_counts = {}
            for meta in self.metadata.values():
                var_type = meta['type']
                type_counts[var_type] = type_counts.get(var_type, 0) + 1

            most_accessed = max(
                self.metadata.items(),
                key=lambda x: x[1]['access_count'],
                default=(None, None)
            )

            return {
                'total_variables': len(self.values),
                'types': type_counts,
                'most_accessed': most_accessed[0] if most_accessed[0] else None,
                'total_accesses': sum(m['access_count'] for m in self.metadata.values())
            }

    def export_to_dict(self) -> Dict[str, Any]:
        with self.lock:
            return {
                'values': self.values.copy(),
                'metadata': self.metadata.copy()
            }

    def import_from_dict(self, data: Dict[str, Any]):
        with self.lock:
            if 'values' in data:
                self.values.update(data['values'])
            if 'metadata' in data:
                self.metadata.update(data['metadata'])
            self.save_to_disk()

    def __str__(self) -> str:
        stats = self.get_statistics()
        return f"ValueStorage: {stats['total_variables']} variables stored"

    def __repr__(self) -> str:
        return f"ValueStorage(storage_path='{self.storage_path}')"