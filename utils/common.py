import os

def touch_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return os.path.normpath(path)
