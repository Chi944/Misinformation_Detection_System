from src.utils.logger import get_logger

logger = get_logger(__name__)


def get_device():
    """
    Detect best available device for PyTorch.

    Checks CUDA first, then MPS (Apple Silicon), then CPU.

    Returns:
        str: 'cuda', 'mps', or 'cpu'
    """
    try:
        import torch

        if torch.cuda.is_available():
            logger.info("CUDA available: %s", torch.cuda.get_device_name(0))
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            logger.info("MPS (Apple Silicon) available")
            return "mps"
    except ImportError:
        pass
    logger.info("Using CPU")
    return "cpu"


def get_tf_device():
    """
    Detect best available device for TensorFlow.

    Returns:
        str: '/GPU:0' if GPU available, else '/CPU:0'
    """
    try:
        import tensorflow as tf

        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            logger.info("TensorFlow GPUs: %s", gpus)
            return "/GPU:0"
    except ImportError:
        pass
    return "/CPU:0"


def set_memory_growth():
    """
    Enable TensorFlow GPU memory growth to prevent OOM errors.

    Should be called before any TensorFlow operations.
    Safe to call even if no GPU is available.
    """
    try:
        import tensorflow as tf

        gpus = tf.config.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            logger.info("Memory growth enabled for %s", gpu)
    except Exception as e:
        logger.warning("Could not set memory growth: %s", e)


def log_device_info():
    """
    Log available hardware information for diagnostics.

    Logs PyTorch and TensorFlow device availability.
    """
    try:
        import torch

        logger.info(
            "PyTorch version=%s CUDA=%s",
            torch.__version__,
            torch.cuda.is_available(),
        )
    except ImportError:
        logger.info("PyTorch not available")
    try:
        import tensorflow as tf

        logger.info("TensorFlow version=%s", tf.__version__)
        gpus = tf.config.list_physical_devices("GPU")
        logger.info("TensorFlow GPUs: %d", len(gpus))
    except ImportError:
        logger.info("TensorFlow not available")
