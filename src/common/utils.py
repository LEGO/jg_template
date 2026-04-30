import logging


def get_logger() -> logging.Logger:
    """Gets a logger with a consistent format and level across the project.

    Configures basic logging if not already set up and adjusts log levels
    for noisy libraries.

    Returns:
        logging.Logger: The configured logger instance.
    """
    logger = logging.getLogger(__name__)
    # Basic configuration if not already configured by the environment / driver
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    logging.getLogger("py4j").setLevel(logging.WARN)
    logging.getLogger("pyspark").setLevel(logging.ERROR)
    logging.getLogger("urllib3").setLevel(logging.WARN)
    logging.getLogger("tensorflow").setLevel(logging.ERROR)
    return logger
