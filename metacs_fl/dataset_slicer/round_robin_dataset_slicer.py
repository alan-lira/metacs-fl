from numpy import array, ndarray
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
        """
        Guarantees that the total number of requested samples per client is used.
        If the client lacks enough data per requested class, redistributes remaining demand to available classes,
        and duplicates proportionally to the local class distribution only if needed.
        """
        if phase == "train":
            class_indices_map = self.class_indices_map_train
            round_robin_offsets = self.round_robin_offsets_train
            x_source, y_source = self.x_train, self.y_train
        elif phase == "test":
            class_indices_map = self.class_indices_map_test
            round_robin_offsets = self.round_robin_offsets_test
            x_source, y_source = self.x_test, self.y_test
        rng = default_rng()
        total_requested = sum(num_examples_per_class.values())
        x_sliced, y_sliced = [], []
        used_per_class = {}
        total_used = 0
        class_deficits = {}
        # Step 1: Fulfill requests without duplication.
        for label_str, requested in num_examples_per_class.items():
            indices = class_indices_map.get(label_str, [])
            total = len(indices)
            if total == 0 or requested <= 0:
                class_deficits[label_str] = requested
                continue
            offset = round_robin_offsets.get(label_str, 0)
            count = min(requested, total)
            selected_indices = [indices[(offset + i) % total] for i in range(count)]
            round_robin_offsets[label_str] = (offset + count) % total
            used_per_class[label_str] = count
            total_used += count
            for idx in selected_indices:
                x_sliced.append(x_source[idx])
                y_sliced.append(y_source[idx])
            if count < requested:
                class_deficits[label_str] = requested - count
        # Step 2: Redistribute deficits to other available local classes (no repeats).
        if class_deficits:
            available_classes = [lbl for lbl in class_indices_map if len(class_indices_map[lbl]) > used_per_class.get(lbl, 0)]
            total_deficit = sum(class_deficits.values())
            if available_classes and total_deficit > 0:
                per_class_extra = total_deficit // len(available_classes)
                extra_remaining = total_deficit % len(available_classes)
                redistribution = {lbl: per_class_extra for lbl in available_classes}
                for i, lbl in enumerate(available_classes):
                    if i < extra_remaining:
                        redistribution[lbl] += 1
                for lbl, add_count in redistribution.items():
                    indices = class_indices_map[lbl]
                    offset = round_robin_offsets.get(lbl, 0)
                    total = len(indices)
                    count = min(add_count, total - offset)
                    if count <= 0:
                        continue
                    selected_indices = [indices[(offset + i) % total] for i in range(count)]
                    round_robin_offsets[lbl] = (offset + count) % total
                    used_per_class[lbl] = used_per_class.get(lbl, 0) + count
                    total_used += count
                    for idx in selected_indices:
                        x_sliced.append(x_source[idx])
                        y_sliced.append(y_source[idx])
        # Step 3: If still not enough, duplicate locally (proportionally to local class distribution).
        if total_used < total_requested:
            deficit = total_requested - total_used
            total_local = sum(len(v) for v in class_indices_map.values())
            if total_local == 0:
                raise ValueError("No local data available to duplicate.")
            # Compute proportional duplication per class.
            duplication_plan = {}
            for lbl, indices in class_indices_map.items():
                proportion = len(indices) / total_local
                duplication_plan[lbl] = int(deficit * proportion)
            # Correct rounding drift.
            assigned = sum(duplication_plan.values())
            if assigned < deficit:
                extra_labels = list(class_indices_map.keys())[:(deficit - assigned)]
                for lbl in extra_labels:
                    duplication_plan[lbl] += 1
            # Duplicate according to plan.
            for lbl, count in duplication_plan.items():
                if count <= 0 or lbl not in class_indices_map:
                    continue
                indices = class_indices_map[lbl]
                chosen = rng.choice(indices, size=count, replace=True)
                for idx in chosen:
                    x_sliced.append(x_source[idx])
                    y_sliced.append(y_source[idx])
                total_used += count
        x_sliced = array(x_sliced, dtype=x_source.dtype)
        y_sliced = array(y_sliced, dtype=y_source.dtype)
        return x_sliced, y_sliced

    def slice_round_robin_balanced(self,
                                   total_examples: int,
                                   phase: str) -> tuple:
        if phase == "train":
            class_labels = list(self.class_indices_map_train.keys())
        elif phase == "test":
            class_labels = list(self.class_indices_map_test.keys())
        num_classes = len(class_labels)
        if num_classes == 0:
            raise ValueError("No classes available for {0}ing phase!".format(phase))
        examples_per_class = total_examples // num_classes
        extra = total_examples % num_classes
        allocation = {label: examples_per_class for label in class_labels}
        for i in range(extra):
            allocation[class_labels[i]] += 1
        # Use strict version.
        x_sliced, y_sliced = self.slice_round_robin_per_class(allocation, phase)
        return x_sliced, y_sliced

    def slice_round_robin_randomly(self,
                                   total_examples: int,
                                   phase: str) -> tuple:
        """
        Randomly cycles through available classes using round-robin fairness,
        but ensures the exact total number of samples (redistribution + proportional duplication if needed).
        """
        if phase == "train":
            class_indices_map = self.class_indices_map_train
            round_robin_offsets = self.round_robin_offsets_train
            x_source, y_source = self.x_train, self.y_train
        elif phase == "test":
            class_indices_map = self.class_indices_map_test
            round_robin_offsets = self.round_robin_offsets_test
            x_source, y_source = self.x_test, self.y_test
        rng = default_rng()
        labels = list(class_indices_map.keys())
        num_labels = len(labels)
        if num_labels == 0:
            raise ValueError("No classes available for {0}ing phase!".format(phase))
        x_sliced, y_sliced = [], []
        total_used = 0
        label_ptr = 0
        # Round-robin selection.
        while total_used < total_examples:
            label = labels[label_ptr]
            indices = class_indices_map[label]
            offset = round_robin_offsets[label]
            if offset < len(indices):
                idx = indices[offset]
                round_robin_offsets[label] += 1
                x_sliced.append(x_source[idx])
                y_sliced.append(y_source[idx])
                total_used += 1
            label_ptr = (label_ptr + 1) % num_labels
            # All classes exhausted → proportional duplication.
            if all(round_robin_offsets[lbl] >= len(class_indices_map[lbl]) for lbl in labels):
                if total_used < total_examples:
                    deficit = total_examples - total_used
                    total_local = sum(len(v) for v in class_indices_map.values())
                    duplication_plan = {}
                    for lbl, indices in class_indices_map.items():
                        proportion = len(indices) / total_local
                        duplication_plan[lbl] = int(deficit * proportion)
                    assigned = sum(duplication_plan.values())
                    if assigned < deficit:
                        extra_labels = labels[:(deficit - assigned)]
                        for lbl in extra_labels:
                            duplication_plan[lbl] += 1
                    for lbl, count in duplication_plan.items():
                        indices = class_indices_map[lbl]
                        chosen = rng.choice(indices, size=count, replace=True)
                        for idx in chosen:
                            x_sliced.append(x_source[idx])
                            y_sliced.append(y_source[idx])
                        total_used += count
                break
        x_sliced = array(x_sliced, dtype=x_source.dtype)
        y_sliced = array(y_sliced, dtype=y_source.dtype)
        return x_sliced, y_sliced
