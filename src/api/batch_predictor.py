# Imports

# > Standard library
import logging
import multiprocessing
import json
import os
import sys
import time
import uuid
from queue import Empty

from typing import List, Tuple

# > Third-party dependencies
import numpy as np
import tensorflow as tf

# Add parent directory to path for imports
current_path = os.path.dirname(os.path.realpath(__file__))
parent_path = os.path.dirname(current_path)
sys.path.append(parent_path)

# > Local imports
from model.management import load_model_from_directory  # noqa
from setup.environment import initialize_strategy  # noqa

# Value used to pad variable-width images into a rectangular batch
PADDING_VALUE = -10.0
# Batch widths are rounded up to a multiple of this, so TensorFlow/Metal only
# ever sees a bounded set of input shapes instead of a new one per batch
WIDTH_BUCKET = 64


def setup_gpu_environment(gpus: str) -> List[tf.config.PhysicalDevice]:
    """
    Configure the GPU environment for TensorFlow.

    Parameters
    ----------
    gpus : str
        IDs of GPUs to be used (comma-separated). Use '-1' for CPU only or
        'all' for all available GPUs.

    Returns
    -------
    List[tf.config.PhysicalDevice]
        List of active GPUs configured for TensorFlow.
    """
    gpu_devices = tf.config.list_physical_devices("GPU")
    logging.info("Available GPUs: %s", gpu_devices)

    if gpus == "-1":
        active_gpus = []
    elif gpus.lower() == "all":
        active_gpus = gpu_devices
    else:
        gpu_indices = gpus.split(",")
        active_gpus = [
            gpu for i, gpu in enumerate(gpu_devices) if str(i) in gpu_indices
        ]

    if active_gpus:
        logging.info("Using GPU(s): %s", active_gpus)
    else:
        logging.info("Using CPU")

    tf.config.set_visible_devices(active_gpus, "GPU")
    for gpu in active_gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    return active_gpus


def get_model_channels(config_path: str) -> int:
    """
    Retrieve the number of input channels for a model from its configuration file.

    Parameters
    ----------
    config_path : str
        Directory path containing the 'config.json' file.

    Returns
    -------
    int
        Number of input channels specified in the configuration.

    Raises
    ------
    FileNotFoundError
        If 'config.json' does not exist in the specified directory.
    ValueError
        If the number of channels is not found in the configuration file.
    """
    config_path = os.path.join(config_path, "config.json")

    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Config file not found in the directory: {config_path}"
        )

    # Load the configuration file
    with open(config_path, "r", encoding="UTF-8") as file:
        config = json.load(file)

    # Extract the number of channels
    # First, check the "model_channels" key, then the "args" key
    num_channels = config.get(
        "model_channels", config.get("args", {}).get("channels", None)
    )
    if num_channels is None:
        raise ValueError("Number of channels not found in the config file.")

    logging.debug("Number of channels retrieved: %d", num_channels)
    return num_channels


def update_channels(model_path: str) -> int:
    """
    Update and retrieve the number of channels from the model's configuration.

    Parameters
    ----------
    model_path : str
        Path to the model directory containing 'config.json'.

    Returns
    -------
    int
        Updated number of input channels.

    Raises
    ------
    FileNotFoundError
        If 'config.json' does not exist in the model directory.
    ValueError
        If the number of channels is not specified in the configuration.
    """

    try:
        num_channels = get_model_channels(model_path)
        logging.debug("New number of channels: %s", num_channels)
        return num_channels
    except Exception as e:
        logging.error("Error updating channels: %s", e)
        raise e


def create_model(
    base_model_dir: str, model_path: str, strategy: tf.distribute.Strategy
) -> (tf.keras.Model, int):
    """
    Load a pre-trained TensorFlow model within the given distribution strategy scope.

    Parameters
    ----------
    base_model_dir : str
        Base directory where models are stored.
    model_path : str
        Relative path to the specific model directory within `base_model_dir`.
    strategy : tf.distribute.Strategy
        TensorFlow distribution strategy for model loading.

    Returns
    -------
    tuple
        A tuple containing the loaded `tf.keras.Model` and the number of input channels.

    Raises
    ------
    Exception
        Propagates any exception that occurs during model loading.
    """
    with strategy.scope():
        try:
            model_location = os.path.join(base_model_dir, model_path)
            model = load_model_from_directory(model_location, compile=False)
            num_channels = update_channels(model_location)
            logging.info(
                "Model '%s' loaded successfully with %d channels.",
                model.name,
                num_channels,
            )
        except Exception as e:
            logging.error("Error loading model from '%s': %s", model_path, e)
            raise
    return model, num_channels


def prepare_image(
    image_bytes: bytes, identifier: str, num_channels: int
) -> np.ndarray:
    """
    Decode and preprocess a single image eagerly.

    Parameters
    ----------
    image_bytes : bytes
        Raw image bytes.
    identifier : str
        Sample identifier, used for error reporting.
    num_channels : int
        Number of channels expected in the image.

    Returns
    -------
    np.ndarray
        Preprocessed image in WHC layout.
    """
    try:
        image = tf.io.decode_image(
            image_bytes, channels=num_channels, expand_animations=False
        )
    except tf.errors.InvalidArgumentError:
        image = tf.zeros([64, 64, num_channels], dtype=tf.float32)
        logging.error("Invalid image for identifier: %s", identifier)

    # Resize and normalize the image
    image = tf.image.resize(image, [64, 99999], preserve_aspect_ratio=True)
    image = tf.cast(image, tf.float32) / 255.0

    # Resize and pad the image
    image = tf.image.resize_with_pad(
        image, 64, tf.shape(image)[1] + 50, method=tf.image.ResizeMethod.BILINEAR
    )

    # Normalize the image
    image = 0.5 - image

    # Transpose the image dimensions if necessary
    image = tf.transpose(image, perm=[1, 0, 2])  # From HWC to WHC

    return image.numpy()


def collect_batch(
    request_queue: multiprocessing.Queue,
    batch_size: int,
    patience: float,
    stop_event: multiprocessing.Event,
    default_model_path: str,
    leftover: tuple = None,
) -> Tuple[list, tuple]:
    """
    Collect a batch of requests that all target the same model.

    The batch is flushed when it is full, when no new requests arrive for
    `patience` seconds, or when a request for a different model comes in.

    Parameters
    ----------
    request_queue : multiprocessing.Queue
        Queue containing incoming requests.
    batch_size : int
        Maximum number of samples per batch.
    patience : float
        Time in seconds to wait for new requests before flushing the batch.
    stop_event : multiprocessing.Event
        Event to signal the collector to stop.
    default_model_path : str
        Model path to use for requests that do not specify one.
    leftover : tuple, optional
        Request carried over from the previous batch after a model switch.

    Returns
    -------
    tuple
        A tuple of (items, leftover): `items` is a list of request tuples for
        a single model; `leftover` is a request targeting a different model
        that should start the next batch, or None.
    """
    items = []
    if leftover is not None:
        items.append(leftover)
    batch_model = items[0][3] if items else None
    last_item_time = time.time()

    while not stop_event.is_set() and len(items) < batch_size:
        try:
            data = request_queue.get(timeout=0.05)
        except Empty:
            if items and time.time() - last_item_time > patience:
                logging.debug(
                    "No new requests for %s seconds. Flushing batch.", patience
                )
                break
            continue

        image_bytes, group_id, identifier, model_path, whitelist = data

        # Ensure model path is always a string (no None)
        if model_path is None:
            model_path = batch_model if batch_model else default_model_path
        item = (image_bytes, group_id, identifier, model_path, whitelist)

        if batch_model is None:
            batch_model = model_path
        elif model_path != batch_model:
            logging.info(
                "Model changed to '%s'. Flushing current batch.", model_path
            )
            return items, item

        items.append(item)
        last_item_time = time.time()

    return items, None


def build_padded_batch(images: List[np.ndarray]) -> np.ndarray:
    """
    Pad variable-width images into a single rectangular batch array.

    Parameters
    ----------
    images : List[np.ndarray]
        Preprocessed images in WHC layout with varying widths.

    Returns
    -------
    np.ndarray
        Batch array of shape [batch, width, height, channels], padded with
        `PADDING_VALUE` up to a bucketed width.
    """
    max_width = max(image.shape[0] for image in images)
    max_width = ((max_width + WIDTH_BUCKET - 1) // WIDTH_BUCKET) * WIDTH_BUCKET

    batch = np.full(
        (len(images), max_width) + images[0].shape[1:],
        PADDING_VALUE,
        dtype=np.float32,
    )
    for i, image in enumerate(images):
        batch[i, : image.shape[0]] = image

    return batch


def pad_whitelists(whitelists: List[List[str]], batch_len: int) -> tf.Tensor:
    """
    Pad whitelists into a rectangular string tensor for the decoder.

    Parameters
    ----------
    whitelists : List[List[str]]
        Whitelist keys per sample.
    batch_len : int
        Number of samples in the batch.

    Returns
    -------
    tf.Tensor
        String tensor of shape [batch, max_whitelist_length], padded with "".
    """
    max_length = max((len(whitelist) for whitelist in whitelists), default=0)
    if max_length == 0:
        return tf.constant("", dtype=tf.string, shape=(batch_len, 0))

    padded = [
        list(whitelist) + [""] * (max_length - len(whitelist))
        for whitelist in whitelists
    ]
    return tf.constant(padded, dtype=tf.string)


def output_prediction_error(
    output_path: str, group_id: str, identifier: str, text: str
):
    """
    Output an error message to a file.

    Parameters
    ----------
    output_path : str
        Base path where prediction outputs should be saved.
    group_id : str
        Group ID of the image.
    identifier : str
        Identifier of the image.
    text : str
        Error message to be saved.
    """

    output_dir = os.path.join(output_path, group_id)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    with open(
        os.path.join(output_dir, identifier + ".error"), "w", encoding="utf-8"
    ) as f:
        f.write(str(text) + "\n")


def predict(
    model: tf.keras.Model, batch_images: tf.Tensor, batch_id: str
) -> np.ndarray:
    """
    Make predictions on a batch of images using the provided model.

    Parameters
    ----------
    model : TensorFlow model
        The model used for making predictions.
    batch_images : tf.Tensor
        A tensor of images for which predictions need to be made.
    batch_id : str
        Unique identifier for the current batch.

    Returns
    -------
    np.ndarray
        An array of predictions made by the model.
    """

    logging.info("Predicting batch of size %d (%s)", len(batch_images), batch_id)
    t1 = time.time()
    encoded_predictions = model.predict_on_batch(batch_images)

    logging.info(
        "Made %d predictions in %.2f seconds (%s)",
        len(encoded_predictions),
        time.time() - t1,
        batch_id,
    )
    logging.debug("Predictions: %s", encoded_predictions)
    return encoded_predictions


def safe_predict(
    model: tf.keras.Model,
    predicted_queue: multiprocessing.Queue,
    batch_images: tf.Tensor,
    batch_info: List[Tuple[str, str]],
    output_path: str,
    batch_id: str,
) -> List[str]:
    """
    Attempt to predict on a batch of images using the provided model. If a
    TensorFlow Out of Memory (OOM) error occurs, the batch is split in half and
    each half is attempted again, recursively. If an OOM error occurs with a
    batch of size 1, the offending image is logged and skipped.

    Parameters
    ----------
    model : TensorFlow model
        The model used for making predictions.
    predicted_queue : multiprocessing.Queue
        Queue where predictions are sent.
    batch_images : tf.Tensor
        A tensor of images for which predictions need to be made.
    batch_info : List of tuples
        A list of tuples containing additional information (e.g., group and
        identifier) for each image in `batch_images`.
    output_path : str
        Path where any output files should be saved.
    batch_id : str
        Unique identifier for the current batch.

    Returns
    -------
    List
        A list of predictions made by the model. If an image causes an OOM
        error, it is skipped, and no prediction is returned for it.
    """

    try:
        return predict(model, batch_images, batch_id)

    except tf.errors.ResourceExhaustedError as e:
        # If the batch size is 1 and still causing OOM, then skip the image and
        # return an empty list
        if len(batch_images) == 1:
            logging.error(
                "OOM error with single image. Skipping image %s.", batch_info[0][1]
            )

            output_prediction_error(output_path, batch_info[0][0], batch_info[0][1], e)
            return []

        logging.warning(
            "OOM error with batch size %s. Splitting batch %s in half and retrying.",
            len(batch_images),
            batch_id,
        )

        # Splitting batch in half
        mid_index = len(batch_images) // 2
        first_half_images = batch_images[:mid_index]
        second_half_images = batch_images[mid_index:]
        first_half_info = batch_info[:mid_index]
        second_half_info = batch_info[mid_index:]

        # Recursive calls for each half
        first_half_predictions = safe_predict(
            model,
            predicted_queue,
            first_half_images,
            first_half_info,
            output_path,
            batch_id,
        )
        second_half_predictions = safe_predict(
            model,
            predicted_queue,
            second_half_images,
            second_half_info,
            output_path,
            batch_id,
        )

        return np.concatenate((first_half_predictions, second_half_predictions))
    except Exception as e:
        logging.error("Error predicting batch %s: %s", batch_id, e)
        for group, identifier in batch_info:
            output_prediction_error(output_path, group, identifier, e)

        return []


def batch_prediction_worker(
    request_queue: multiprocessing.Queue,
    predicted_queue: multiprocessing.Queue,
    base_model_dir: str,
    initial_model_path: str,
    error_output_path: str,
    stop_event: multiprocessing.Event,
    gpus: str = "0",
    batch_size: int = 32,
    patience: int = 1,
):
    """
    Worker process for performing batch predictions on images.

    Batching is done directly from the request queue in plain Python instead
    of through a repeatedly recreated `tf.data.Dataset.from_generator`
    pipeline: recreating that pipeline after every idle period leaked native
    memory that was never reclaimed for the lifetime of the process.

    Parameters
    ----------
    request_queue : multiprocessing.Queue
        Queue from which raw images are fetched.
    predicted_queue : multiprocessing.Queue
        Queue to which predictions are put.
    base_model_dir : str
        Base directory where models are stored.
    initial_model_path : str
        Initial model path relative to `base_model_dir`.
    error_output_path : str
        Base path where prediction errors should be saved.
    stop_event : multiprocessing.Event
        Event to signal the worker to stop processing.
    gpus : str, optional
        IDs of GPUs to be used (comma-separated). Use '-1' for CPU only or 'all'
        for all available GPUs. Default is '0'.
    batch_size : int, optional
        Number of samples per batch. Default is 32.
    patience : int, optional
        Time in seconds to wait for new requests before yielding the current batch. Default is 1.

    Side Effects
    ------------
    - Logs various messages regarding the batch processing status.
    - Loads and reloads models as needed.
    - Puts prediction results into `predicted_queue`.
    """
    logging.info("Batch prediction worker started")

    # Configure GPU environment
    active_gpus = setup_gpu_environment(gpus)
    strategy = initialize_strategy(use_float32=False, active_gpus=active_gpus)

    # Load the initial model
    current_model_path_holder = [initial_model_path]
    model, num_channels = create_model(
        base_model_dir, current_model_path_holder[0], strategy
    )

    leftover = None
    try:
        while not stop_event.is_set():
            batch, leftover = collect_batch(
                request_queue,
                batch_size,
                patience,
                stop_event,
                current_model_path_holder[0],
                leftover,
            )

            if not batch:
                continue

            # Check for model updates
            model_path = batch[0][3]
            if model_path != current_model_path_holder[0]:
                logging.info(
                    "Model switch detected. Replacing old model '%s' with model '%s'.",
                    current_model_path_holder[0],
                    model_path,
                )
                current_model_path_holder[0] = model_path
                model, num_channels = create_model(
                    base_model_dir, model_path, strategy
                )
                logging.debug("Model '%s' loaded successfully.", model_path)

            # Preprocess and pad the batch
            images = [
                prepare_image(image_bytes, identifier, num_channels)
                for image_bytes, _, identifier, _, _ in batch
            ]
            batch_images = build_padded_batch(images)

            batch_groups = [group_id for _, group_id, _, _, _ in batch]
            batch_identifiers = [identifier for _, _, identifier, _, _ in batch]
            batch_whitelists = pad_whitelists(
                [whitelist for _, _, _, _, whitelist in batch], len(batch)
            )

            # Perform predictions
            batch_id = str(uuid.uuid4())
            encoded_predictions = safe_predict(
                model,
                predicted_queue,
                batch_images,
                list(zip(batch_groups, batch_identifiers)),
                error_output_path,
                batch_id,
            )
            logging.debug("Predictions made for batch %s", batch_id)
            predicted_queue.put(
                (
                    encoded_predictions,
                    tf.constant(batch_groups, dtype=tf.string),
                    tf.constant(batch_identifiers, dtype=tf.string),
                    model_path,
                    batch_id,
                    batch_whitelists,
                )
            )
    except Exception as e:
        logging.error("Error in batch prediction worker: %s", e)
        raise e
    finally:
        logging.info("Batch prediction worker stopped")
