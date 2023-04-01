import threading

class SingletonPtn(object):
    _instance_lock = threading.Lock()

    def __init__(self):
        pass

    def __new__(cls, *args, **kwargs):
        if not hasattr(SingletonPtn, "_instance"):
            with SingletonPtn._instance_lock:
                if not hasattr(SingletonPtn, "_instance"):
                    SingletonPtn._instance = object.__new__(cls)
        return SingletonPtn._instance

    