import sys
from os import devnull, environ, getpid

# Suppress TensorFlow C++ log messages (redirecting stderr to null).
environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
sys.stderr = open(devnull, "w")

from contractions import fix
from emoji import demojize
from flwr.common import NDArray
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner, IidPartitioner, PathologicalPartitioner
from keras.applications.densenet import preprocess_input as densenet121_preprocess_input
from keras.applications.efficientnet import preprocess_input as efficientnet_preprocess_input
from keras.applications.efficientnet_v2 import preprocess_input as efficientnet_v2_preprocess_input
from keras.applications.mobilenet_v2 import preprocess_input as mobilenet_v2_preprocess_input
from keras.applications.resnet import preprocess_input as resnet50_preprocess_input
from keras.applications.vgg16 import preprocess_input as vgg16_preprocess_input
from nltk.corpus import stopwords
from numpy import array, asarray, empty, int64, ndarray, int32
from numpy.random import normal
from pathlib import Path
from pickle import dump as pickle_dump, load as pickle_load
from PIL import Image
from random import sample
from re import sub
from requests import get
from tensorflow import expand_dims
from tensorflow.image import resize
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from time import perf_counter, sleep
from tqdm import tqdm
from urllib.request import urlretrieve
from zipfile import ZipFile


def get_images_count(dataset_folder: Path) -> int:
    return sum(1 for sub_path in dataset_folder.rglob("*") if sub_path.suffix in {".gif", ".jpg", ".jpeg", ".png"})


def get_images_attributes(dataset_folder: Path) -> tuple:
    first_folder = [sub_path for sub_path in dataset_folder.iterdir() if sub_path.is_dir()][0]
    first_image_from_folder = [sub_path for sub_path in first_folder.rglob("*")
                               if sub_path.is_file() and sub_path.suffix in {".gif", ".jpg", ".jpeg", ".png"}][0]
    image = Image.open(fp=first_image_from_folder)
    width, height = image.size
    depth = len(image.getbands())
    return width, height, depth


def get_classes_distribution(y: NDArray) -> dict:
    classes_distribution = {}
    for index in range(len(y)):
        val = y[index]
        if isinstance(val, ndarray):
            val = val.item() if val.size == 1 else val[0]
        y_label = str(int(val))
        classes_distribution[y_label] = classes_distribution.get(y_label, 0) + 1
    sorted_keys = sorted(classes_distribution.keys(), key=lambda x: (len(x), x))
    classes_distribution = {k: classes_distribution[k] for k in sorted_keys}
    return classes_distribution


def _load_x_y_for_multiclass_image_dataset(dataset_root_folder: Path,
                                           phase: str) -> tuple:
    dataset_folder = dataset_root_folder.joinpath(phase)
    images_count = get_images_count(dataset_folder)
    width, height, depth = get_images_attributes(dataset_folder)
    derived_x_shape = (images_count, height, width, depth)
    derived_y_shape = (images_count, 1)
    x = empty(shape=derived_x_shape, dtype="uint8")
    y = empty(shape=derived_y_shape, dtype="uint8")
    index = 0
    for sub_path in dataset_folder.iterdir():
        if sub_path.is_dir():
            for inner_sub_path in sub_path.rglob("*"):
                if inner_sub_path.is_file() and inner_sub_path.suffix in {".gif", ".jpg", ".jpeg", ".png"}:
                    x[index] = Image.open(fp=inner_sub_path)
                    y[index] = int(sub_path.stem)
                    index += 1
    return x, y


def _load_local_dataset(local_dataset_settings: dict) -> tuple:
    # Get the necessary attributes.
    dataset_root_folder = Path(local_dataset_settings["dataset_root_folder"])
    dataset_type = local_dataset_settings["dataset_type"]
    # Initialize x_train, y_train, x_test, and y_test.
    x_train = y_train = x_test = y_test = None
    match dataset_type:
        case "multi_class_image_classification":
            # Load x_train and y_train.
            x_train, y_train = _load_x_y_for_multiclass_image_dataset(dataset_root_folder, "train")
            # Load x_test and y_test.
            x_test, y_test = _load_x_y_for_multiclass_image_dataset(dataset_root_folder, "test")
    # Return the loaded dataset (x_train, y_train, x_test, and y_test).
    return x_train, y_train, x_test, y_test


def instantiate_fds(federated_dataset_settings: dict) -> FederatedDataset:
    # Get the necessary attributes.
    dataset = federated_dataset_settings["dataset"]
    subset = federated_dataset_settings["subset"]
    dataset_partitioner = federated_dataset_settings["dataset_partitioner"]
    num_partitions = federated_dataset_settings["num_partitions"]
    # Set the necessary keys.
    train_partitioner_key = ""
    test_partitioner_key = ""
    match dataset:
        case "uoft-cs/cifar10":
            train_partitioner_key = "train"
            test_partitioner_key = "test"
        case "uoft-cs/cifar100":
            train_partitioner_key = "train"
            test_partitioner_key = "test"
        case "ylecun/mnist":
            train_partitioner_key = "train"
            test_partitioner_key = "test"
        case "zalando-datasets/fashion_mnist":
            train_partitioner_key = "train"
            test_partitioner_key = "test"
        case "zh-plus/tiny-imagenet":
            train_partitioner_key = "train"
            test_partitioner_key = "valid"
        case "benjamin-paine/imagenet-1k":
            train_partitioner_key = "train"
            test_partitioner_key = "validation"
        case "ufldl-stanford/svhn":
            train_partitioner_key = "train"
            test_partitioner_key = "test"
        case "flwrlabs/cinic10":
            train_partitioner_key = "train"
            test_partitioner_key = "test"
        case "adilbekovich/Sentiment140Twitter":
            train_partitioner_key = "train"
            test_partitioner_key = "test"
    partitioners = {}
    training_dataset_partitioner = None
    test_dataset_partitioner = None
    match dataset_partitioner:
        case "IidPartitioner":
            # Set the training dataset partitioner.
            training_dataset_partitioner = IidPartitioner(num_partitions=num_partitions)
            # Set the test dataset partitioner.
            test_dataset_partitioner = IidPartitioner(num_partitions=num_partitions)
        case "DirichletPartitioner":
            # Get the training dataset settings.
            training_dataset_min_partition_size = federated_dataset_settings["training_dataset_min_partition_size"]
            training_dataset_alpha = federated_dataset_settings["training_dataset_alpha"]
            training_dataset_partition_by = federated_dataset_settings["training_dataset_partition_by"]
            training_dataset_self_balancing = federated_dataset_settings["training_dataset_self_balancing"]
            training_dataset_shuffle = federated_dataset_settings["training_dataset_shuffle"]
            training_dataset_seed = federated_dataset_settings["training_dataset_seed"]
            # Get the test dataset settings.
            test_dataset_min_partition_size = federated_dataset_settings["test_dataset_min_partition_size"]
            test_dataset_alpha = federated_dataset_settings["test_dataset_alpha"]
            test_dataset_partition_by = federated_dataset_settings["test_dataset_partition_by"]
            test_dataset_self_balancing = federated_dataset_settings["test_dataset_self_balancing"]
            test_dataset_shuffle = federated_dataset_settings["test_dataset_shuffle"]
            test_dataset_seed = federated_dataset_settings["test_dataset_seed"]
            # Set the training dataset partitioner.
            training_dataset_partitioner = DirichletPartitioner(num_partitions=num_partitions,
                                                                partition_by=training_dataset_partition_by,
                                                                alpha=training_dataset_alpha,
                                                                min_partition_size=training_dataset_min_partition_size,
                                                                self_balancing=training_dataset_self_balancing,
                                                                shuffle=training_dataset_shuffle,
                                                                seed=training_dataset_seed)
            # Set the test dataset partitioner.
            test_dataset_partitioner = DirichletPartitioner(num_partitions=num_partitions,
                                                            partition_by=test_dataset_partition_by,
                                                            alpha=test_dataset_alpha,
                                                            min_partition_size=test_dataset_min_partition_size,
                                                            self_balancing=test_dataset_self_balancing,
                                                            shuffle=test_dataset_shuffle,
                                                            seed=test_dataset_seed)
        case "PathologicalPartitioner":
            # Get the training dataset settings.
            training_dataset_partition_by = federated_dataset_settings["training_dataset_partition_by"]
            training_num_classes_per_partition = federated_dataset_settings["training_num_classes_per_partition"]
            training_class_assignment_mode = federated_dataset_settings["training_class_assignment_mode"]
            training_dataset_shuffle = federated_dataset_settings["training_dataset_shuffle"]
            training_dataset_seed = federated_dataset_settings["training_dataset_seed"]
            # Get the test dataset settings.
            test_dataset_partition_by = federated_dataset_settings["test_dataset_partition_by"]
            test_num_classes_per_partition = federated_dataset_settings["test_num_classes_per_partition"]
            test_class_assignment_mode = federated_dataset_settings["test_class_assignment_mode"]
            test_dataset_shuffle = federated_dataset_settings["test_dataset_shuffle"]
            test_dataset_seed = federated_dataset_settings["test_dataset_seed"]
            # Set the training dataset partitioner.
            training_dataset_partitioner = PathologicalPartitioner(num_partitions=num_partitions,
                                                                   partition_by=training_dataset_partition_by,
                                                                   num_classes_per_partition=training_num_classes_per_partition,
                                                                   class_assignment_mode=training_class_assignment_mode,
                                                                   shuffle=training_dataset_shuffle,
                                                                   seed=training_dataset_seed)
            # Set the test dataset partitioner.
            test_dataset_partitioner = PathologicalPartitioner(num_partitions=num_partitions,
                                                               partition_by=test_dataset_partition_by,
                                                               num_classes_per_partition=test_num_classes_per_partition,
                                                               class_assignment_mode=test_class_assignment_mode,
                                                               shuffle=test_dataset_shuffle,
                                                               seed=test_dataset_seed)
    # Update the dictionary of partitioners.
    partitioners.update({train_partitioner_key: training_dataset_partitioner,
                         test_partitioner_key: test_dataset_partitioner})
    # Instantiate the FederatedDataset object.
    fds = FederatedDataset(dataset=dataset,
                           subset=subset,
                           partitioners=partitioners)
    # Return the FederatedDataset object.
    return fds


def _load_federated_dataset(client_id: int,
                            federated_dataset_settings: dict,
                            fds: FederatedDataset) -> tuple:
    # Get the necessary attributes.
    dataset = federated_dataset_settings["dataset"]
    # Set the necessary keys.
    x_field_key = ""
    y_field_key = ""
    train_split_key = ""
    test_split_key = ""
    match dataset:
        case "uoft-cs/cifar10":
            x_field_key = "img"
            y_field_key = "label"
            train_split_key = "train"
            test_split_key = "test"
        case "uoft-cs/cifar100":
            x_field_key = "img"
            y_field_key = "coarse_label"  # TODO: Improve the definition of the y_field key.
            train_split_key = "train"
            test_split_key = "test"
        case "ylecun/mnist":
            x_field_key = "image"
            y_field_key = "label"
            train_split_key = "train"
            test_split_key = "test"
        case "zalando-datasets/fashion_mnist":
            x_field_key = "image"
            y_field_key = "label"
            train_split_key = "train"
            test_split_key = "test"
        case "zh-plus/tiny-imagenet":
            x_field_key = "image"
            y_field_key = "label"
            train_split_key = "train"
            test_split_key = "valid"
        case "benjamin-paine/imagenet-1k":
            x_field_key = "image"
            y_field_key = "label"
            train_split_key = "train"
            test_split_key = "validation"
        case "ufldl-stanford/svhn":
            x_field_key = "image"
            y_field_key = "label"
            train_split_key = "train"
            test_split_key = "test"
        case "flwrlabs/cinic10":
            x_field_key = "image"
            y_field_key = "label"
            train_split_key = "train"
            test_split_key = "test"
        case "adilbekovich/Sentiment140Twitter":
            x_field_key = "text"
            y_field_key = "label"
            train_split_key = "train"
            test_split_key = "test"
    # Get the client's partitions (based on its id).
    partition_train = fds.load_partition(client_id, train_split_key)
    partition_train.set_format("numpy")
    partition_test = fds.load_partition(client_id, test_split_key)
    partition_test.set_format("numpy")
    # Load x_train and y_train.
    x_train, y_train = partition_train[x_field_key], partition_train[y_field_key]
    # Load x_test and y_test.
    x_test, y_test = partition_test[x_field_key], partition_test[y_field_key]
    # Return the loaded dataset (x_train, y_train, x_test, and y_test).
    return x_train, y_train, x_test, y_test


def _reshape_images(x: NDArray,
                    new_shape: tuple) -> NDArray:
    x_reshaped = []
    for index in range(0, len(x)):
        xi_copy = x[index].copy()
        xi_copy.resize(new_shape)
        x_reshaped.append(xi_copy)
    x_reshaped = array(x_reshaped)
    return x_reshaped


def _build_or_load_shared_tokenizer(client_id: int,
                                    fds: FederatedDataset,
                                    federated_dataset_settings: dict,
                                    vocab_size: int,
                                    root_output_folder: Path,
                                    timeout: int = 1800) -> Tokenizer:
    root_output_folder.mkdir(parents=True, exist_ok=True)
    tokenizer_file = root_output_folder / "shared_tokenizer.pkl"
    lock_path = tokenizer_file.with_suffix(".lock")
    # If the tokenizer already exists, load it immediately.
    if tokenizer_file.exists():
        with tokenizer_file.open("rb") as f:
            return pickle_load(f)
    try:
        # Try to acquire exclusive lock (atomic file creation).
        with lock_path.open("x") as lock_file:
            lock_file.write("locked")
        print("[Client {0}] Lock acquired, building shared tokenizer...".format(client_id))
        # Only the first client builds the tokenizer.
        all_texts = []
        num_partitions = federated_dataset_settings["num_partitions"]
        # Collect all text data from every partition.
        for cid in range(num_partitions):
            partition = fds.load_partition(cid, "train")
            partition.set_format("numpy")
            all_texts.extend(partition["text"])
        tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
        tokenizer.fit_on_texts(all_texts)
        # Save the shared tokenizer.
        with tokenizer_file.open("wb") as f:
            pickle_dump(tokenizer, f)
        print("[Client {0}] Shared tokenizer saved at {1}".format(client_id, tokenizer_file))
        return tokenizer
    except FileExistsError:
        # Another client is currently building the tokenizer.
        print("[Client {0}] Waiting for shared tokenizer to be ready...".format(client_id))
        waited = 0
        while not tokenizer_file.exists():
            sleep(2)
            waited += 2
            if waited > timeout:
                raise TimeoutError("[Client {0}] Timeout after {1}s waiting for tokenizer.".format(client_id, timeout))
        with tokenizer_file.open("rb") as f:
            tokenizer = pickle_load(f)
        print("[Client {0}] Loaded shared tokenizer after waiting.".format(client_id))
        return tokenizer
    finally:
        # If this process created the lock, remove it after completion.
        if lock_path.exists():
            lock_path.unlink()


def _normalize_sentiment140_text(text: str) -> str:
    """Normalize a single tweet, based on the paper 'Federated Learning for Sentiment Analysis in Presence of Non-IID Data:
    Sensitivity of Deep Learning Models' (Gholamiangonabadi & Grolinger, 2024)"""
    # Replace emojis with descriptive words (e.g., ":)" → "smile").
    text = demojize(text, delimiters=(" ", " "))
    # Expand contractions (e.g., "don't" → "do not").
    text = fix(text)
    # Lowercase all characters.
    text = text.lower()
    # Remove URLs, mentions, and hashtags (keep hashtag words).
    text = sub(r"http\S+|www\S+", "", text)
    text = sub(r"@\w+", "", text)
    text = sub(r"#", "", text)
    # Remove punctuation, numbers, and non-alphabetic characters.
    text = sub(r"[^a-z\s]", " ", text)
    # Collapse multiple spaces.
    text = sub(r"\s+", " ", text).strip()
    # Remove stopwords except "no", "not", "none".
    stopwords_list = set(stopwords.words("english")) - {"no", "not", "none"}
    words = [w for w in text.split() if w not in stopwords_list]
    return " ".join(words)


def _pre_process_sentiment140_text_dataset(texts: NDArray,
                                           labels: NDArray,
                                           vocab_size: int,
                                           max_length: int,
                                           tokenizer: Tokenizer | None = None) -> tuple:
    # Ensure list input.
    texts = list(texts) if not isinstance(texts, list) else texts
    # Filter out None or empty strings.
    valid_indices, valid_texts = [], []
    for i, text in enumerate(texts):
        if text is not None and str(text).strip():
            valid_indices.append(i)
            valid_texts.append(str(text).strip())
    if not valid_texts:
        print("Warning: No valid texts found after filtering")
        empty = array([])
        return (empty, empty if labels is not None else None, tokenizer)
    # Normalization step.
    normalized_texts = [_normalize_sentiment140_text(t) for t in valid_texts]
    # Tokenization step.
    if tokenizer is None:
        tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
        tokenizer.fit_on_texts(normalized_texts)
    # Convert texts to integer sequences.
    sequences = tokenizer.texts_to_sequences(normalized_texts)
    # Remove empty sequences.
    non_empty_sequences, non_empty_indices = [], []
    for i, seq in enumerate(sequences):
        if len(seq) > 0:
            non_empty_sequences.append(seq)
            non_empty_indices.append(valid_indices[i])
    if not non_empty_sequences:
        print("Warning: All sequences empty after tokenization")
        empty = array([])
        return (empty, empty if labels is not None else None, tokenizer)
    # Pad sequences.
    padded_sequences = pad_sequences(non_empty_sequences, maxlen=max_length, padding="post")
    # Get the labels.
    binary_labels = None
    if labels is not None:
        labels_array = array(labels)
        valid_labels = labels_array[non_empty_indices]
        # Dataset already uses 0/1, no remapping needed
        binary_labels = valid_labels.astype(int32)
    return padded_sequences, binary_labels, tokenizer


# Progress bar hook.
class DownloadProgressBar(tqdm):

    def update_to(self,
                  b: int = 1,
                  bsize: int = 1,
                  tsize: int = None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def _ensure_glove_embeddings_are_available(embedding_dim: int = 100,
                                           glove_source: str = "twitter",
                                           glove_embeddings_folder: Path = "glove_embeddings") -> Path:
    url = "https://nlp.stanford.edu/data/glove.{0}.27B.zip".format(glove_source)
    glove_embeddings_folder.mkdir(parents=True, exist_ok=True)
    glove_dir = glove_embeddings_folder / "glove.{0}.27B".format(glove_source)
    zip_path = glove_embeddings_folder / "glove.{0}.27B.zip".format(glove_source)
    txt_path = glove_dir / "glove.{0}.27B.{1}d.txt".format(glove_source, embedding_dim)
    flag_path = glove_embeddings_folder / ".glove_download_in_progress.flag"
    # If embeddings already available, just return.
    if txt_path.exists():
        return txt_path
    # If another process is downloading, wait for it.
    wait_time = 0
    while flag_path.exists():
        if wait_time % 5 == 0:
            print("[PID: {0}] Another process is downloading embeddings, waiting...".format(getpid()))
        sleep(2)
        wait_time += 2
        if wait_time > 1800:
            raise TimeoutError("Waited too long for embedding download to finish! ({0} seconds)".format(wait_time))
    # Start own download with exclusive flag.
    try:
        flag_path.touch(exist_ok=False)
    except FileExistsError:
        # Another process beat us to it, wait for completion.
        return _ensure_glove_embeddings_are_available(embedding_dim, glove_source, glove_embeddings_folder)
    try:
        # Double-check if someone else already completed it.
        if txt_path.exists():
            flag_path.unlink(missing_ok=True)
            return txt_path
        # Download.
        print("[PID: {0}] Downloading GloVe embeddings from {1} ...".format(getpid(), url))
        with DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc="Downloading GloVe") as t:
            urlretrieve(url, zip_path, reporthook=t.update_to)
        # Extract.
        print("PID: [{0}] Extracting {1} ...".format(getpid(), zip_path))
        with ZipFile(zip_path, "r") as zf:
            zf.extractall(glove_dir)
        # Ready message.
        print("[{0}] GloVe embeddings ready at {1}".format(getpid(), txt_path))
    finally:
        # Remove flag.
        flag_path.unlink(missing_ok=True)
    return txt_path


def _load_glove_embeddings(tokenizer: Tokenizer,
                           embedding_dim: int,
                           glove_embedding_file: Path,
                           global_vocab_size: int) -> ndarray:
    if not glove_embedding_file.exists():
        raise FileNotFoundError("GloVe file not found at {0}".format(glove_embedding_file))
    # Fix the matrix size to the global vocabulary.
    embedding_matrix = normal(scale=0.6, size=(global_vocab_size, embedding_dim))
    print("Loading GloVe embeddings from: {0}".format(glove_embedding_file))
    glove_embeddings = {}
    with open(glove_embedding_file, encoding="utf8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = asarray(values[1:], dtype="float32")
            glove_embeddings[word] = vector
    # Map words in tokenizer to embedding indices.
    for word, idx in tokenizer.word_index.items():
        if idx < global_vocab_size:
            vector = glove_embeddings.get(word)
            if vector is not None:
                embedding_matrix[idx] = vector
    print("GloVe embedding matrix shape: {0}".format(embedding_matrix.shape))
    return embedding_matrix


def _pre_process_image_dataset(dataset: str,
                               model_settings: dict,
                               x_train: NDArray,
                               x_test: NDArray) -> tuple:
    # Set the list of custom CNNs.
    custom_cnns = ["Custom_CNN_CIFAR-10", "Custom_CNN_CIFAR-100_Fine_Labels", "Custom_CNN_CIFAR-100_Coarse_Labels",
                   "Custom_CNN_MNIST", "Custom_CNN_FashionMNIST", "Custom_CNN_SVHN", "Custom_CNN_CINIC-10",
                   "Custom_CNN_TinyImageNet"]
    # Set the list of literature models that accepts minimum input_shape of 32x32.
    models_min_32_input_size = ["EfficientNetB0", "EfficientNetV2L", "MobileNetV2", "VGG16", "ResNet50", "DenseNet121"]
    # Get the necessary attributes.
    model_provider = model_settings["provider"]
    model_provider_settings = model_settings[model_provider]
    model_name = model_provider_settings["model_name"]
    # Reshape the data instances, if needed.
    match dataset:
        case "flwrlabs/cinic10":
            new_shape = (32, 32, 3)
            x_train = _reshape_images(x_train, new_shape)
            x_test = _reshape_images(x_test, new_shape)
        case "zh-plus/tiny-imagenet":
            new_shape = (64, 64, 3)
            x_train = _reshape_images(x_train, new_shape)
            x_test = _reshape_images(x_test, new_shape)
    # Pre-process the dataset.
    if model_provider == "Keras":
        match model_name:
            case custom_cnn if custom_cnn in custom_cnns:
                x_train = x_train / 255.0
                x_test = x_test / 255.0
            case model_min_32_input_size if model_min_32_input_size in models_min_32_input_size:
                if "mnist" in dataset.lower():
                    # Resize images from 28x28 to 32x32 (minimum required input size of models).
                    x_train = resize(expand_dims(x_train, axis=-1), (32, 32))
                    x_test = resize(expand_dims(x_test, axis=-1), (32, 32))
            case "EfficientNetB0":
                x_train = efficientnet_preprocess_input(x=x_train)
                x_test = efficientnet_preprocess_input(x=x_test)
            case "EfficientNetV2L":
                x_train = efficientnet_v2_preprocess_input(x=x_train)
                x_test = efficientnet_v2_preprocess_input(x=x_test)
            case "MobileNetV2":
                x_train = mobilenet_v2_preprocess_input(x=x_train)
                x_test = mobilenet_v2_preprocess_input(x=x_test)
            case "VGG16":
                x_train = vgg16_preprocess_input(x=x_train)
                x_test = vgg16_preprocess_input(x=x_test)
            case "ResNet50":
                x_train = resnet50_preprocess_input(x=x_train)
                x_test = resnet50_preprocess_input(x=x_test)
            case "DenseNet121":
                x_train = densenet121_preprocess_input(x=x_train)
                x_test = densenet121_preprocess_input(x=x_test)
    # Return the pre-processed dataset (x_train, x_test).
    return x_train, x_test


def load_dataset(client_id: int,
                 loading_approach: str,
                 local_dataset_settings: dict,
                 federated_dataset_settings: dict,
                 model_settings: dict,
                 fds: FederatedDataset,
                 root_output_folder: Path) -> dict:
    # Initialize the dataset loading dict.
    dataset_loading_dict = {}
    # Get the model specific settings (can be necessary for pre-processing of dataset).
    model_provider = model_settings["provider"]
    model_provider_settings = model_settings[model_provider]
    model_name = model_provider_settings["model_name"]
    model_provider_specific_settings = model_settings[model_name]
    # Start the dataset loading duration timer.
    dataset_loading_duration_start = perf_counter()
    # Initialize x_train, y_train, x_test, and y_test.
    x_train = y_train = x_test = y_test = None
    # Initialize the dataset name.
    dataset = None
    match loading_approach:
        case "Local":
            x_train, y_train, x_test, y_test = _load_local_dataset(local_dataset_settings)
            dataset = local_dataset_settings["dataset"]
        case "FederatedDataset":
            x_train, y_train, x_test, y_test = _load_federated_dataset(client_id, federated_dataset_settings, fds)
            dataset = federated_dataset_settings["dataset"]
    # Handle dataset preprocessing separately.
    if dataset == "adilbekovich/Sentiment140Twitter":
        # Pre-process the text dataset (sentiment140).
        vocab_size = model_provider_specific_settings["vocab_size"]
        max_length = model_provider_specific_settings["max_length"]
        # Build or load the "shared" tokenizer.
        tokenizer = _build_or_load_shared_tokenizer(client_id, fds, federated_dataset_settings, vocab_size, root_output_folder)
        # Pre-process the training data (fits the tokenizer).
        x_train, y_train, _ = _pre_process_sentiment140_text_dataset(x_train,
                                                                     y_train,
                                                                     vocab_size,
                                                                     max_length,
                                                                     tokenizer=tokenizer)
        # Pre-process the test data (reuse the same tokenizer).
        x_test, y_test, _ = _pre_process_sentiment140_text_dataset(x_test,
                                                                   y_test,
                                                                   vocab_size,
                                                                   max_length,
                                                                   tokenizer=tokenizer)
        # Load GloVe embeddings (aligned with the tokenizer) if configured.
        embedding_matrix = None
        use_glove_embedding = model_provider_specific_settings["use_glove_embedding"]
        if use_glove_embedding:
            embedding_dim = model_provider_specific_settings["embedding_dim"]
            glove_source = model_provider_specific_settings["glove_source"]
            glove_embeddings_folder = Path(model_provider_specific_settings["glove_embeddings_folder"])
            glove_embedding_file = _ensure_glove_embeddings_are_available(embedding_dim, glove_source, glove_embeddings_folder)
            embedding_matrix = _load_glove_embeddings(tokenizer, embedding_dim, glove_embedding_file, vocab_size)
        dataset_loading_dict.update({"embedding_matrix": embedding_matrix})
    elif dataset in ["uoft-cs/cifar10", "uoft-cs/cifar100", "ylecun/mnist", "zalando-datasets/fashion_mnist",
                     "zh-plus/tiny-imagenet", "benjamin-paine/imagenet-1k", "ufldl-stanford/svhn", "flwrlabs/cinic10"]:
        # Pre-process the image dataset.
        x_train, x_test = _pre_process_image_dataset(dataset, model_settings, x_train, x_test)
    # Get the dataset load duration.
    dataset_loading_duration = perf_counter() - dataset_loading_duration_start
    # Return the dataset loading dict.
    dataset_loading_dict.update({"x_train": x_train,
                                 "y_train": y_train,
                                 "x_test": x_test,
                                 "y_test": y_test,
                                 "dataset_loading_duration": dataset_loading_duration})
    return dataset_loading_dict


def get_task_assignment_capacities(x_train: NDArray,
                                   x_test: NDArray,
                                   task_assignment_capacities_settings: dict,
                                   samples_per_task: int) -> tuple:
    # Get the necessary attributes.
    task_assignment_capacities_train = list(task_assignment_capacities_settings["task_assignment_capacities_train"])
    task_assignment_capacities_test = list(task_assignment_capacities_settings["task_assignment_capacities_test"])
    lower_bound = task_assignment_capacities_settings["lower_bound"]
    upper_bound = task_assignment_capacities_settings["upper_bound"]
    step = task_assignment_capacities_settings["step"]
    # Scale the bounds for training.
    if upper_bound == "client_capacity":
        upper_bound_train = len(x_train) // samples_per_task
    else:
        upper_bound_train = upper_bound // samples_per_task
    lower_bound_train = lower_bound // samples_per_task
    step_train = max(1, step // samples_per_task)
    # Scale the bounds for testing.
    if upper_bound == "client_capacity":
        upper_bound_test = len(x_test) // samples_per_task
    else:
        upper_bound_test = upper_bound // samples_per_task
    lower_bound_test = lower_bound // samples_per_task
    step_test = max(1, step // samples_per_task)
    # Compute train capacities.
    if not task_assignment_capacities_train:
        task_assignment_capacities_train = [lower_bound_train, upper_bound_train]
        task_assignment_capacities_train.extend(list(range(lower_bound_train, upper_bound_train + 1, step_train)))
    # Compute test capacities.
    if not task_assignment_capacities_test:
        task_assignment_capacities_test = [lower_bound_test, upper_bound_test]
        task_assignment_capacities_test.extend(list(range(lower_bound_test, upper_bound_test + 1, step_test)))
    # Remove duplicates and ensure sorted order.
    task_assignment_capacities_train = sorted(set(task_assignment_capacities_train))
    task_assignment_capacities_test = sorted(set(task_assignment_capacities_test))
    return task_assignment_capacities_train, task_assignment_capacities_test


def _slice_dataset_per_class(x: NDArray,
                             y: NDArray,
                             num_examples_per_class: dict) -> tuple:
    # Initialize the sliced dataset.
    x_sliced = []
    y_sliced = []
    # Map the x,y values.
    x_y_map = {}
    for index in range(0, len(y)):
        y_label = None
        if type(y[index]) == ndarray:
            y_label = str(y[index][0])
        elif type(y[index]) == int64:
            y_label = str(y[index])
        if y_label not in x_y_map:
            x_y_map.update({y_label: [index]})
        else:
            x_y_map[y_label].append(index)
    # Initialize the x, y sliced map.
    x_y_sliced_map = {}
    # Fill the x, y sliced map.
    for k, v in num_examples_per_class.items():
        # Randomly choose v examples of class k.
        k_sample = sample(x_y_map[k], v)
        x_y_sliced_map.update({k: k_sample})
    # Fill the sliced dataset.
    for y_label, x_indices in x_y_sliced_map.items():
        for x_index in x_indices:
            x_sliced.append(x[x_index])
            y_sliced.append(y[x_index])
    # Convert the sliced dataset into Numpy arrays.
    x_sliced = array(x_sliced, dtype=x.dtype)
    y_sliced = array(y_sliced, dtype=y.dtype)
    # Return the sliced dataset.
    return x_sliced, y_sliced


def _slice_dataset_random_manner(x: NDArray,
                                 y: NDArray,
                                 num_examples: int) -> tuple:
    # Initialize the sliced dataset.
    x_sliced = []
    y_sliced = []
    # Map the x,y values.
    x_y_map = {}
    for index in range(0, len(y)):
        y_label = None
        if type(y[index]) == ndarray:
            y_label = str(y[index][0])
        elif type(y[index]) == int64:
            y_label = str(y[index])
        if y_label not in x_y_map:
            x_y_map.update({y_label: [index]})
        else:
            x_y_map[y_label].append(index)
    # Initialize the x, y sliced map.
    x_y_sliced_map = {}
    # Fill the x, y sliced map.
    while True:
        if num_examples == 0:
            break
        y_label = sample(list(x_y_map.keys()), 1)[0]
        if len(x_y_map[y_label]) > 0:
            sampled_x_index = sample(x_y_map[y_label], 1)[0]
            x_y_map[y_label].remove(sampled_x_index)
            if y_label not in x_y_sliced_map:
                x_y_sliced_map.update({y_label: [sampled_x_index]})
            else:
                x_y_sliced_map[y_label].append(sampled_x_index)
            num_examples -= 1
            if num_examples == 0:
                break
    # Fill the sliced dataset.
    for y_label, x_indices in x_y_sliced_map.items():
        for x_index in x_indices:
            x_sliced.append(x[x_index])
            y_sliced.append(y[x_index])
    # Convert the sliced dataset into Numpy arrays.
    x_sliced = array(x_sliced, dtype=x.dtype)
    y_sliced = array(y_sliced, dtype=y.dtype)
    # Return the sliced dataset.
    return x_sliced, y_sliced


def _slice_dataset_balanced_manner(x: NDArray,
                                   y: NDArray,
                                   num_examples: int) -> tuple:
    # Initialize the sliced dataset.
    x_sliced = []
    y_sliced = []
    # Map the x,y values.
    x_y_map = {}
    for index in range(0, len(y)):
        y_label = None
        if type(y[index]) == ndarray:
            y_label = str(y[index][0])
        elif type(y[index]) == int64:
            y_label = str(y[index])
        if y_label not in x_y_map:
            x_y_map.update({y_label: [index]})
        else:
            x_y_map[y_label].append(index)
    # Initialize the x, y sliced map.
    x_y_sliced_map = {}
    # Fill the x, y sliced map.
    while True:
        if num_examples == 0:
            break
        for y_label, x_indices in x_y_map.items():
            if len(x_y_map[y_label]) > 0:
                sampled_x_index = sample(x_y_map[y_label], 1)[0]
                x_y_map[y_label].remove(sampled_x_index)
                if y_label not in x_y_sliced_map:
                    x_y_sliced_map.update({y_label: [sampled_x_index]})
                else:
                    x_y_sliced_map[y_label].append(sampled_x_index)
                num_examples -= 1
                if num_examples == 0:
                    break
    # Fill the sliced dataset.
    for y_label, x_indices in x_y_sliced_map.items():
        for x_index in x_indices:
            x_sliced.append(x[x_index])
            y_sliced.append(y[x_index])
    # Convert the sliced dataset into Numpy arrays.
    x_sliced = array(x_sliced, dtype=x.dtype)
    y_sliced = array(y_sliced, dtype=y.dtype)
    # Return the sliced dataset.
    return x_sliced, y_sliced


def slice_dataset(phase_config: dict,
                  x: NDArray,
                  y: NDArray) -> tuple:
    total_examples = len(x)
    examples_per_class_to_use_key = "examples_per_class_to_use"
    matched_examples_per_class_to_use = next(((key, phase_config[key]) for key in phase_config.keys()
                                              if examples_per_class_to_use_key in key), None)
    if matched_examples_per_class_to_use:
        num_examples_per_class_to_use = phase_config[matched_examples_per_class_to_use[0]]
        num_examples_per_class_to_use = num_examples_per_class_to_use.split("|")
        num_examples_per_class_to_use = {str(elem).split("=")[0]: int(str(elem).split("=")[1])
                                         for elem in num_examples_per_class_to_use}
        num_examples_to_use = sum(num_examples_per_class_to_use.values())
        if num_examples_to_use < total_examples:
            x, y = _slice_dataset_per_class(x, y, num_examples_per_class_to_use)
    else:
        examples_to_use_key = "examples_to_use"
        matched_examples_to_use = next(((key, phase_config[key]) for key in phase_config.keys()
                                        if examples_to_use_key in key), None)
        if matched_examples_to_use:
            num_examples_to_use = phase_config[matched_examples_to_use[0]]
            if num_examples_to_use < total_examples:
                if "dataset_slice_approach" in phase_config:
                    dataset_slice_approach = phase_config["dataset_slice_approach"]
                    match dataset_slice_approach:
                        case "Random_Slice":
                            x, y = _slice_dataset_random_manner(x, y, num_examples_to_use)
                        case "Balanced_Slice":
                            x, y = _slice_dataset_balanced_manner(x, y, num_examples_to_use)
                else:
                    x, y = _slice_dataset_random_manner(x, y, num_examples_to_use)
    return x, y
