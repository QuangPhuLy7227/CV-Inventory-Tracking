import time

class FPS:
    def __init__(self):
        self.last = time.time()
        self.value = 0.0

    def tick(self):
        now = time.time()
        dt = now - self.last
        self.last = now
        if dt > 0:
            self.value = 1.0 / dt
        return self.value