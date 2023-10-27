import time


class MyTimer():

    def __init__(self, name=""):
        self.name = name
        self.ini_time = time.time()

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        elapsed = time.time() - self.ini_time
        if self.name:
            header = f"[{self.name}] "
        else:
            header = ""
        print(f"{header}Elapsed: {elapsed:.3f} s.")
