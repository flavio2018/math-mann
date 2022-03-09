from datetime import datetime
import logging


def configure_logging(loglevel, run_name):
    # assuming loglevel is bound to the string value obtained from the
    # command line argument. Convert to upper case to allow the user to
    # specify --log=DEBUG or --log=debug
    numeric_level = getattr(logging, loglevel.upper(), None)

    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % loglevel)

    logging.basicConfig(
        filename=f"../logs/{run_name}.log",
        level=numeric_level,
        format='%(levelname)s:%(filename)s:%(message)s',
        filemode="w")


def get_str_timestamp():
    return datetime.now().strftime("%Y%B%d_%H-%M-%S")
