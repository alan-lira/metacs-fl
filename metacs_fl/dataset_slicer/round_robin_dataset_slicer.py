from numpy import array, int64, ndarray, unique
from numpy.random import default_rng


class RoundRobinDatasetSlicer:

    def __init__(self,
                 x_train: ndarray,
                 y_train: ndarray,
                 x_test: ndarray,
                 y_test: ndarray) -> None:
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.class_indices_map_train = self._build_class_index_map(self.y_train)
        self.class_indices_map_test = self._build_class_index_map(self.y_test)
        self.round_robin_offsets_train = {k: 0 for k in self.class_indices_map_train}
        self.round_robin_offsets_test = {k: 0 for k in self.class_indices_map_test}

    @staticmethod
    def _build_class_index_map(y: ndarray) -> dict:
        rng = default_rng()
        class_index_map = {}
        for idx, label in enumerate(y):
            label_str = str(label[0]) if isinstance(label, ndarray) else str(label)
            class_index_map.setdefault(label_str, []).append(idx)
        for indices in class_index_map.values():
            rng.shuffle(indices)
        return class_index_map

    def slice_round_robin_per_class(self,
                                    num_examples_per_class: dict,
                                    phase: str) -> tuple:
        x_sliced = []
        y_sliced = []
        class_indices_map = {}
        round_robin_offsets = {}
        if phase == "train":
            x_sliced, y_sliced = [], []
            class_indices_map = self.class_indices_map_train
            round_robin_offsets = self.round_robin_offsets_train
        elif phase == "test":
            x_sliced, y_sliced = [], []
            class_indices_map = self.class_indices_map_test
            round_robin_offsets = self.round_robin_offsets_test
        for label_str, count in num_examples_per_class.items():
            indices = class_indices_map[label_str]
            offset = round_robin_offsets[label_str]
            total = len(indices)
            selected_indices = [indices[(offset + i) % total] for i in range(count)]
            round_robin_offsets[label_str] = (offset + count) % total
            for idx in selected_indices:
                if phase == "train":
                    x_sliced.append(self.x_train[idx])
                    y_sliced.append(self.y_train[idx])
                elif phase == "test":
                    x_sliced.append(self.x_test[idx])
                    y_sliced.append(self.y_test[idx])
        x_sliced = array(x_sliced, dtype=self.x_train.dtype)
        y_sliced = array(y_sliced, dtype=self.y_train.dtype)
        return x_sliced, y_sliced

    def slice_round_robin_balanced(self,
                                   total_examples: int,
                                   phase: str) -> tuple:
        class_labels = []
        if phase == "train":
            class_labels = list(self.class_indices_map_train.keys())
        elif phase == "test":
            class_labels = list(self.class_indices_map_test.keys())
        num_classes = len(class_labels)
        examples_per_class = total_examples // num_classes
        extra = total_examples % num_classes
        allocation = {label: examples_per_class for label in class_labels}
        for i in range(extra):
            allocation[class_labels[i]] += 1
        return self.slice_round_robin_per_class(allocation, phase)

    def slice_round_robin_randomly(self,
                                   total_examples: int,
                                   phase: str) -> tuple:
        x_sliced = []
        y_sliced = []
        class_indices_map = []
        round_robin_offsets = []
        if phase == "train":
            class_indices_map = self.class_indices_map_train
            round_robin_offsets = self.round_robin_offsets_train
        elif phase == "test":
            class_indices_map = self.class_indices_map_test
            round_robin_offsets = self.round_robin_offsets_test
        labels = list(class_indices_map.keys())
        num_labels = len(labels)
        label_ptr = 0
        total_examples_available = sum(len(v) - round_robin_offsets[k] for k, v in class_indices_map.items())
        if total_examples > total_examples_available:
            raise ValueError("Requested {0} examples, but only {1} are available (phase: {2})."
                             .format(total_examples, total_examples_available, phase))
        while total_examples > 0:
            label = labels[label_ptr]
            indices = class_indices_map[label]
            offset = round_robin_offsets[label]
            if offset < len(indices):
                idx = indices[offset]
                round_robin_offsets[label] += 1
                if phase == "train":
                    x_sliced.append(self.x_train[idx])
                    y_sliced.append(self.y_train[idx])
                else:
                    x_sliced.append(self.x_test[idx])
                    y_sliced.append(self.y_test[idx])
                total_examples -= 1
            label_ptr = (label_ptr + 1) % num_labels
            if all(round_robin_offsets[lbl] >= len(class_indices_map[lbl]) for lbl in labels):
                break
        x_sliced = array(x_sliced, dtype=self.x_train.dtype if phase == "train" else self.x_test.dtype)
        y_sliced = array(y_sliced, dtype=self.y_train.dtype if phase == "train" else self.y_test.dtype)
        return x_sliced, y_sliced
