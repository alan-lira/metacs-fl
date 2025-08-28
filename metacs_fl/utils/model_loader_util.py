import sys
from os import devnull, environ

# Suppress TensorFlow C++ log messages (redirecting stderr to null).
environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
sys.stderr = open(devnull, "w")

from flwr.common import Parameters
from io import BytesIO
from keras import layers, Model, Sequential
from keras.applications import DenseNet121, EfficientNetB0, EfficientNetV2L, MobileNetV2, ResNet50, VGG16
from keras.losses import Loss, SparseCategoricalCrossentropy
from keras.metrics import Metric, SparseCategoricalAccuracy
from keras.optimizers import Adam, Optimizer, SGD
from keras.optimizers.schedules import CosineDecay, ExponentialDecay
from keras.saving import load_model as keras_load_model, save_model as keras_save_model
from numpy import load, save, savez
from pathlib import Path


def load_custom_cnn_cifar_10() -> Model:
    # Define a custom CNN for the CIFAR-10 dataset (FedCS Paper):
    #  Specifically, our model consisted of six 3 × 3 convolution layers (32, 32, 64, 64,
    #  128, 128 channels, each of which was activated by ReLU and batch normalized,
    #  and every two of which were followed by 2 × 2 max pooling)
    #  followed by three fully-connected layers (382 and 192 units with ReLU activation and
    #  another 10 units activated by soft-max).
    model = Sequential()
    # Input Layer.
    model.add(layers.Input(shape=(32, 32, 3)))
    # First Block: Conv(32) -> BatchNorm -> ReLU.
    model.add(layers.Conv2D(32, (3, 3), padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    # Second Block: Conv(32) -> BatchNorm -> ReLU.
    model.add(layers.Conv2D(32, (3, 3), padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    # 2 x 2 Max Pooling.
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    # Third Block: Conv(64) -> BatchNorm -> ReLU.
    model.add(layers.Conv2D(64, (3, 3), padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    # Fourth Block: Conv(64) -> BatchNorm -> ReLU.
    model.add(layers.Conv2D(64, (3, 3), padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    # 2 x 2 Max Pooling.
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    # Fifth Block: Conv(128) -> BatchNorm -> ReLU.
    model.add(layers.Conv2D(128, (3, 3), padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    # Sixth Block: Conv(128) -> BatchNorm -> ReLU.
    model.add(layers.Conv2D(128, (3, 3), padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    # 2 x 2 Max Pooling.
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    # Flatten the output from the convolutional layers.
    model.add(layers.Flatten())
    # First Fully Connected Layer: 382 units, ReLU activation.
    model.add(layers.Dense(382, activation="relu"))
    # Second Fully Connected Layer: 192 units, ReLU activation.
    model.add(layers.Dense(192, activation="relu"))
    # Third Fully Connected Layer: 10 units (output layer), softmax activation for multi-class classification.
    model.add(layers.Dense(10, activation="softmax"))
    # Return the model.
    return model


def load_custom_cnn_cifar_100_fine_labels() -> Model:
    # Define a custom CNN for the CIFAR-100 dataset (100 fine labels).
    model = Sequential()
    # Input Layer.
    model.add(layers.Input(shape=(32, 32, 3)))
    # First Convolutional Block.
    model.add(layers.Conv2D(32, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    # Second Convolutional Block.
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    # Third Convolutional Block.
    model.add(layers.Conv2D(128, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    # Flatten the output for the fully connected layers.
    model.add(layers.Flatten())
    # Fully Connected Layers.
    model.add(layers.Dense(128, activation="relu"))
    # Dropout layer to reduce overfitting.
    model.add(layers.Dropout(0.5))
    # Output layer for 100 classes (softmax activation for multi-class classification).
    model.add(layers.Dense(100, activation="softmax"))
    # Return the model.
    return model


def load_custom_cnn_cifar_100_coarse_labels() -> Model:
    # Define a custom CNN for the CIFAR-100 dataset (20 coarse labels).
    model = Sequential()
    # Input Layer.
    model.add(layers.Input(shape=(32, 32, 3)))
    # First Convolutional Block.
    model.add(layers.Conv2D(32, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    # Second Convolutional Block.
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    # Third Convolutional Block.
    model.add(layers.Conv2D(128, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    # Flatten the output for the fully connected layers.
    model.add(layers.Flatten())
    # Fully Connected Layers.
    model.add(layers.Dense(128, activation="relu"))
    # Dropout layer to reduce overfitting.
    model.add(layers.Dropout(0.5))
    # Output layer for 20 classes (softmax activation for multi-class classification).
    model.add(layers.Dense(20, activation="softmax"))
    # Return the model.
    return model


def load_custom_cnn_mnist() -> Model:
    # Define a custom CNN for the MNIST dataset.
    model = Sequential()
    # Input Layer.
    model.add(layers.Input(shape=(28, 28, 1)))
    # First Convolutional Block.
    model.add(layers.Conv2D(16, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    # Second Convolutional Block.
    model.add(layers.Conv2D(32, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    # Flatten the output for the fully connected layers.
    model.add(layers.Flatten())
    # Dropout layer to reduce overfitting.
    model.add(layers.Dropout(0.3))
    # Additional dense layer to improve feature extraction.
    model.add(layers.Dense(64, activation="relu"))
    # Output layer for 10 classes (softmax activation for multi-class classification).
    model.add(layers.Dense(10, activation="softmax"))
    # Return the model.
    return model


def load_custom_cnn_fashion_mnist() -> Model:
    # Define a custom CNN for the FashionMNIST dataset.
    model = Sequential()
    # Input Layer.
    model.add(layers.Input(shape=(28, 28, 1)))
    # First Convolutional Block.
    model.add(layers.Conv2D(32, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    # Second Convolutional Block.
    model.add(layers.Conv2D(128, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    # Flatten the output for the fully connected layers.
    model.add(layers.Flatten())
    # Dropout layer to reduce overfitting.
    model.add(layers.Dropout(0.5))
    # Output layer for 10 classes (softmax activation for multi-class classification).
    model.add(layers.Dense(10, activation="softmax"))
    # Return the model.
    return model


def load_custom_cnn_tiny_imagenet() -> Model:
    # Define a custom CNN for the Tiny ImageNet dataset.
    model = Sequential()
    # Input Layer.
    model.add(layers.Input(shape=(64, 64, 3)))
    # First Convolutional Block.
    model.add(layers.Conv2D(32, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    # Second Convolutional Block.
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    # Third Convolutional Block.
    model.add(layers.Conv2D(128, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    # Flatten the output for the fully connected layers.
    model.add(layers.Flatten())
    # Fully Connected Layers.
    model.add(layers.Dense(512, activation="relu"))
    # Dropout layer to reduce overfitting.
    model.add(layers.Dropout(0.5))
    # Output layer for 200 classes (softmax activation for multi-class classification).
    model.add(layers.Dense(200, activation="softmax"))
    # Return the model.
    return model


def load_custom_cnn_svhn() -> Model:
    # Define a custom CNN for the SVHN dataset.
    model = Sequential()
    # Input Layer.
    model.add(layers.Input(shape=(32, 32, 3)))
    # First Convolutional Block.
    model.add(layers.Conv2D(32, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    # Second Convolutional Block.
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    # Flatten the output for the fully connected layers.
    model.add(layers.Flatten())
    # Fully Connected Layers.
    model.add(layers.Dense(128, activation="relu"))
    # Dropout layer to reduce overfitting.
    model.add(layers.Dropout(0.5))
    # Output layer for 10 classes (softmax activation for multi-class classification).
    model.add(layers.Dense(10, activation="softmax"))
    # Return the model.
    return model


def load_custom_cnn_cinic_10() -> Model:
    # Define a custom CNN for the CINIC-10 dataset.
    model = Sequential()
    # Input Layer.
    model.add(layers.Input(shape=(32, 32, 3)))
    # First Convolutional Block.
    model.add(layers.Conv2D(32, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    # Second Convolutional Block.
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    # Third Convolutional Block.
    model.add(layers.Conv2D(128, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    # Flatten the output for the fully connected layers.
    model.add(layers.Flatten())
    # Fully Connected Layers.
    model.add(layers.Dense(128, activation="relu"))
    # Dropout layer to reduce overfitting.
    model.add(layers.Dropout(0.5))
    # Output layer for 10 classes (softmax activation for multi-class classification).
    model.add(layers.Dense(10, activation="softmax"))
    # Return the model.
    return model


def load_efficientnet_b0(model_provider_specific_settings: dict) -> Model:
    # Instantiate the Kera's EfficientNetB0 model.
    model = EfficientNetB0(input_shape=model_provider_specific_settings["input_shape"],
                           include_top=model_provider_specific_settings["include_top"],
                           weights=model_provider_specific_settings["weights"],
                           input_tensor=model_provider_specific_settings["input_tensor"],
                           pooling=model_provider_specific_settings["pooling"],
                           classes=model_provider_specific_settings["classes"],
                           classifier_activation=model_provider_specific_settings["classifier_activation"])
    # Return the model.
    return model


def load_efficientnet_v2l(model_provider_specific_settings: dict) -> Model:
    # Instantiate the Kera's EfficientNetV2L model.
    model = EfficientNetV2L(input_shape=model_provider_specific_settings["input_shape"],
                            include_top=model_provider_specific_settings["include_top"],
                            weights=model_provider_specific_settings["weights"],
                            input_tensor=model_provider_specific_settings["input_tensor"],
                            pooling=model_provider_specific_settings["pooling"],
                            classes=model_provider_specific_settings["classes"],
                            classifier_activation=model_provider_specific_settings["classifier_activation"])
    # Return the model.
    return model


def load_mobilenet_v2(model_provider_specific_settings: dict) -> Model:
    # Instantiate the Kera's MobileNetV2 model.
    model = MobileNetV2(input_shape=model_provider_specific_settings["input_shape"],
                        alpha=model_provider_specific_settings["alpha"],
                        include_top=model_provider_specific_settings["include_top"],
                        weights=model_provider_specific_settings["weights"],
                        input_tensor=model_provider_specific_settings["input_tensor"],
                        pooling=model_provider_specific_settings["pooling"],
                        classes=model_provider_specific_settings["classes"],
                        classifier_activation=model_provider_specific_settings["classifier_activation"])
    # Return the model.
    return model


def load_vgg_16(model_provider_specific_settings: dict) -> Model:
    # Instantiate the Kera's VGG16 model.
    model = VGG16(input_shape=model_provider_specific_settings["input_shape"],
                  include_top=model_provider_specific_settings["include_top"],
                  weights=model_provider_specific_settings["weights"],
                  input_tensor=model_provider_specific_settings["input_tensor"],
                  pooling=model_provider_specific_settings["pooling"],
                  classes=model_provider_specific_settings["classes"],
                  classifier_activation=model_provider_specific_settings["classifier_activation"])
    # Return the model.
    return model


def load_resnet_50(model_provider_specific_settings: dict) -> Model:
    # Instantiate the Kera's ResNet50 model.
    model = ResNet50(input_shape=model_provider_specific_settings["input_shape"],
                     include_top=model_provider_specific_settings["include_top"],
                     weights=model_provider_specific_settings["weights"],
                     input_tensor=model_provider_specific_settings["input_tensor"],
                     pooling=model_provider_specific_settings["pooling"],
                     classes=model_provider_specific_settings["classes"],
                     classifier_activation=model_provider_specific_settings["classifier_activation"])
    # Return the model.
    return model


def load_densenet_121(model_provider_specific_settings: dict) -> Model:
    # Instantiate the Kera's DenseNet121 model.
    model = DenseNet121(input_shape=model_provider_specific_settings["input_shape"],
                        include_top=model_provider_specific_settings["include_top"],
                        weights=model_provider_specific_settings["weights"],
                        input_tensor=model_provider_specific_settings["input_tensor"],
                        pooling=model_provider_specific_settings["pooling"],
                        classes=model_provider_specific_settings["classes"],
                        classifier_activation=model_provider_specific_settings["classifier_activation"])
    # Return the model.
    return model


def load_optimizer(model_settings: dict,
                   learning_rate_schedule_settings: dict) -> Optimizer:
    # Get the necessary attributes.
    model_provider = model_settings["provider"]
    model_provider_settings = model_settings[model_provider]
    optimizer_name = model_provider_settings["optimizer_name"]
    optimizer_settings = model_settings[optimizer_name]
    learning_rate = optimizer_settings["learning_rate"]
    enable_learning_rate_schedule = learning_rate_schedule_settings["enable_learning_rate_schedule"]
    if enable_learning_rate_schedule:
        learning_rate_schedule_name = learning_rate_schedule_settings["learning_rate_schedule_name"]
        match learning_rate_schedule_name:
            case "CosineDecay":
                initial_learning_rate = learning_rate_schedule_settings["initial_learning_rate"]
                decay_steps = learning_rate_schedule_settings["decay_steps"]
                alpha = learning_rate_schedule_settings["alpha"]
                name = learning_rate_schedule_settings["name"]
                warmup_target = learning_rate_schedule_settings["warmup_target"]
                warmup_steps = learning_rate_schedule_settings["warmup_steps"]
                learning_rate = CosineDecay(initial_learning_rate=initial_learning_rate,
                                            decay_steps=decay_steps,
                                            alpha=alpha,
                                            name=name,
                                            warmup_target=warmup_target,
                                            warmup_steps=warmup_steps)
            case "ExponentialDecay":
                initial_learning_rate = learning_rate_schedule_settings["initial_learning_rate"]
                decay_steps = learning_rate_schedule_settings["decay_steps"]
                decay_rate = learning_rate_schedule_settings["decay_rate"]
                staircase = learning_rate_schedule_settings["staircase"]
                name = learning_rate_schedule_settings["name"]
                learning_rate = ExponentialDecay(initial_learning_rate=initial_learning_rate,
                                                 decay_steps=decay_steps,
                                                 decay_rate=decay_rate,
                                                 staircase=staircase,
                                                 name=name)
    # Initialize the optimizer.
    optimizer = None
    if model_provider == "Keras":
        match optimizer_name:
            case "Adam":
                # Instantiate the Kera's Adam optimizer.
                optimizer = Adam(learning_rate=learning_rate,
                                 beta_1=optimizer_settings["beta_1"],
                                 beta_2=optimizer_settings["beta_2"],
                                 epsilon=optimizer_settings["epsilon"],
                                 amsgrad=optimizer_settings["amsgrad"],
                                 weight_decay=optimizer_settings["weight_decay"],
                                 clipnorm=optimizer_settings["clipnorm"],
                                 clipvalue=optimizer_settings["clipvalue"],
                                 global_clipnorm=optimizer_settings["global_clipnorm"],
                                 use_ema=optimizer_settings["use_ema"],
                                 ema_momentum=optimizer_settings["ema_momentum"],
                                 ema_overwrite_frequency=optimizer_settings["ema_overwrite_frequency"],
                                 loss_scale_factor=optimizer_settings["loss_scale_factor"],
                                 gradient_accumulation_steps=optimizer_settings["gradient_accumulation_steps"])
            case "SGD":
                # Instantiate the Kera's SGD optimizer (Stochastic Gradient Descent).
                optimizer = SGD(learning_rate=learning_rate,
                                momentum=optimizer_settings["momentum"],
                                nesterov=optimizer_settings["nesterov"],
                                name=optimizer_settings["optimizer_name"])
    # Return the optimizer.
    return optimizer


def load_loss_function(model_settings: dict) -> Loss:
    # Get the necessary attributes.
    model_provider = model_settings["provider"]
    model_provider_settings = model_settings[model_provider]
    loss_name = model_provider_settings["loss_name"]
    loss_settings = model_settings[loss_name]
    # Initialize the loss.
    loss = None
    if model_provider == "Keras":
        match loss_name:
            case "SparseCategoricalCrossentropy":
                # Instantiate the Kera's SparseCategoricalCrossentropy loss function.
                loss = SparseCategoricalCrossentropy(from_logits=loss_settings["from_logits"],
                                                     ignore_class=loss_settings["ignore_class"],
                                                     reduction=loss_settings["reduction"],
                                                     name=loss_settings["loss_name"])
    # Return the loss function.
    return loss


def load_metrics(model_settings: dict) -> list[Metric]:
    # Get the necessary attributes.
    model_provider = model_settings["provider"]
    model_provider_settings = model_settings[model_provider]
    metrics = model_provider_settings["metrics"]
    for index, metric in enumerate(metrics):
        if model_provider == "Keras":
            match metric:
                case "sparse_categorical_accuracy":
                    # Instantiate the Kera's SparseCategoricalAccuracy metric.
                    metrics[index] = SparseCategoricalAccuracy()
    # Return the list of metrics.
    return metrics


def load_model(model_settings: dict,
               learning_rate_schedule_settings: dict) -> tuple:
    model_provider = model_settings["provider"]
    model_provider_settings = model_settings[model_provider]
    model_name = model_provider_settings["model_name"]
    model_provider_specific_settings = model_settings[model_name]
    optimizer = load_optimizer(model_settings, learning_rate_schedule_settings)
    loss_function = load_loss_function(model_settings)
    metrics = load_metrics(model_settings)
    model = None
    metrics_names = None
    if model_provider == "Keras":
        match model_name:
            case "Custom_CNN_CIFAR-10":
                model = load_custom_cnn_cifar_10()
            case "Custom_CNN_CIFAR-100_Fine_Labels":
                model = load_custom_cnn_cifar_100_fine_labels()
            case "Custom_CNN_CIFAR-100_Coarse_Labels":
                model = load_custom_cnn_cifar_100_coarse_labels()
            case "Custom_CNN_MNIST":
                model = load_custom_cnn_mnist()
            case "Custom_CNN_FashionMNIST":
                model = load_custom_cnn_fashion_mnist()
            case "Custom_CNN_TinyImageNet":
                model = load_custom_cnn_tiny_imagenet()
            case "Custom_CNN_SVHN":
                model = load_custom_cnn_svhn()
            case "Custom_CNN_CINIC-10":
                model = load_custom_cnn_cinic_10()
            case "EfficientNetB0":
                model = load_efficientnet_b0(model_provider_specific_settings)
            case "EfficientNetV2L":
                model = load_efficientnet_v2l(model_provider_specific_settings)
            case "MobileNetV2":
                model = load_mobilenet_v2(model_provider_specific_settings)
            case "VGG16":
                model = load_vgg_16(model_provider_specific_settings)
            case "ResNet50":
                model = load_resnet_50(model_provider_specific_settings)
            case "DenseNet121":
                model = load_densenet_121(model_provider_specific_settings)
        # Compile the Kera's model.
        loss_weights = model_provider_settings["loss_weights"]
        weighted_metrics = model_provider_settings["weighted_metrics"]
        run_eagerly = model_provider_settings["run_eagerly"]
        steps_per_execution = model_provider_settings["steps_per_execution"]
        jit_compile = model_provider_settings["jit_compile"]
        auto_scale_loss = model_provider_settings["auto_scale_loss"]
        model.compile(optimizer=optimizer,
                      loss=loss_function,
                      loss_weights=loss_weights,
                      metrics=metrics,
                      weighted_metrics=weighted_metrics,
                      run_eagerly=run_eagerly,
                      steps_per_execution=steps_per_execution,
                      jit_compile=jit_compile,
                      auto_scale_loss=auto_scale_loss)
        # Get the model's metrics names.
        metrics_names = [metric.name for metric in vars(model)["_compile_metrics"]._user_metrics]
    return model, metrics_names


def serialize_model_weights(model: Model) -> bytes:
    buffer = BytesIO()
    savez(buffer, *model.get_weights())
    model_weights_serialized = buffer.getvalue()
    return model_weights_serialized


def deserialize_model_weights(model_weights_serialized: bytes) -> list:
    buffer = BytesIO(model_weights_serialized)
    npz = load(buffer, allow_pickle=True)
    model_weights_deserialized = [npz[k] for k in npz]
    return model_weights_deserialized


def cast_model_parameters(model: Model) -> Parameters:
    tensors = []
    for weight in model.get_weights():
        buffer = BytesIO()
        save(buffer, weight, allow_pickle=False)
        tensor_bytes = buffer.getvalue()
        tensors.append(tensor_bytes)
    flower_parameters = Parameters(tensors=tensors, tensor_type="numpy.ndarray")
    return flower_parameters


def load_model_from_file(model_file: Path) -> Model:
    model = keras_load_model(filepath=model_file, compile=True, safe_mode=True)
    return model


def save_model_to_file(model: Model,
                       model_file: Path) -> None:
    model_file.parent.mkdir(exist_ok=True, parents=True)
    keras_save_model(model=model, filepath=model_file, overwrite=True)
