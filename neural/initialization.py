import math
import numpy as np

class Initialization:
    def initialize(shape: tuple[int], rng: np.random.Generator|None = None) -> np.array:
        pass

class HeInitialization:
    def initialize(shape: tuple[int], rng: np.random.Generator|None = None) -> np.array:
        rng = rng or np.random.default_rng()
        return rng.normal(0, np.sqrt(2/shape[0]), shape)

class RandomInitialization:
    def initialize(shape: tuple[int], rng: np.random.Generator|None = None) -> np.array:
        rng = rng or np.random.default_rng()
        return rng.random(shape)
