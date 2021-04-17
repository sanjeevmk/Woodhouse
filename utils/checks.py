import os
import errno

def check_file_exists(file_path):
    """
    If given path doesn't exist, raise an exception and exit.

    :param file_path: Path to file
    :return: If file doesn't exist, raises Exception
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(errno.ENOENT,os.strerror(errno.ENOENT),file_path)