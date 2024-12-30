import os
import numpy as np
import mmap
import threading
from collections import defaultdict
import portalocker

class DataManager:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.data_dir = os.path.join(base_dir, "model_data")
        self.data_catalog = defaultdict(lambda: defaultdict(list))
        self.memory_maps = {}
        self.file_locks = {}
        self.load_data_catalog()

    def load_data_catalog(self):
        for data_type in ['raw_data', 'cleaned_data', 'preprocessed_data', 'validation_data', 'features']:
            type_dir = os.path.join(self.data_dir, data_type)
            if os.path.exists(type_dir):
                for action in os.listdir(type_dir):
                    action_dir = os.path.join(type_dir, action)
                    if os.path.isdir(action_dir):
                        for f in os.listdir(action_dir):
                            if f.endswith('.npy'):
                                file_path = os.path.join(action_dir, f)
                                self.data_catalog[data_type][action].append(file_path)

    def load_data(self, data_type, action, sample_identifier):
        if action in self.data_catalog[data_type]:
            if isinstance(sample_identifier, int):
                # Load by index
                if sample_identifier < len(self.data_catalog[data_type][action]):
                    file_path = self.data_catalog[data_type][action][sample_identifier]
                else:
                    return None
            else:
                # Load by filename
                file_path = os.path.join(self.data_dir, data_type, action, sample_identifier)
                if file_path not in self.data_catalog[data_type][action]:
                    return None

            lock = self.file_locks.setdefault(file_path, threading.Lock())
            with lock:
                try:
                    with portalocker.Lock(file_path, 'rb', timeout=60) as f:
                        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                        data = np.load(mm)
                        mm.close()
                        return data
                except Exception as e:
                    print(f"Error loading data from {file_path}: {str(e)}")
                    return None
        return None

    def save_data(self, data_type, action, data, file_name):
        action_dir = os.path.join(self.data_dir, data_type, action)
        os.makedirs(action_dir, exist_ok=True)
        file_path = os.path.join(action_dir, file_name)
        lock = self.file_locks.setdefault(file_path, threading.Lock())
        with lock:
            with portalocker.Lock(file_path, 'wb', timeout=60) as f:
                np.save(f, data)
        if file_path not in self.data_catalog[data_type][action]:
            self.data_catalog[data_type][action].append(file_path)

    def get_actions(self, data_type):
        return list(self.data_catalog[data_type].keys())

    def get_sample_count(self, data_type, action):
        return len(self.data_catalog[data_type][action])

    def get_data_summary(self):
        summary = {}
        for data_type in self.data_catalog.keys():
            summary[data_type] = {}
            for action in self.data_catalog[data_type]:
                sample_count = len(self.data_catalog[data_type][action])
                if sample_count > 0:
                    sample = self.load_data(data_type, action, 0)
                    shape = sample.shape if sample is not None else None
                    summary[data_type][action] = {'count': sample_count, 'shape': shape}
        return summary

    def clear_cached_features(self, version):
        features_dir = os.path.join(self.data_dir, 'features')
        if os.path.exists(features_dir):
            for action in os.listdir(features_dir):
                action_dir = os.path.join(features_dir, action)
                if os.path.isdir(action_dir):
                    for f in os.listdir(action_dir):
                        if f.endswith('.npy') and not f.endswith(f'_v{version}.npy'):
                            os.remove(os.path.join(action_dir, f))
        self.load_data_catalog()  # Reload the catalog after clearing old features