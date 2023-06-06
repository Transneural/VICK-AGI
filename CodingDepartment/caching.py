import os
import pickle
import hashlib

CACHE_DIR = 'cache/'

def cache_code(code, filename):
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
    
    with open(os.path.join(CACHE_DIR, filename), 'wb') as f:
        pickle.dump(code, f)

def load_cached_code(filename):
    try:
        with open(os.path.join(CACHE_DIR, filename), 'rb') as f:
            return pickle.load(f)
    except:
        return None

def cache_function(func):
    def wrapper(*args, **kwargs):
        cache_key = hashlib.md5(pickle.dumps((func.__name__, args, kwargs))).hexdigest()
        cached_result = load_cached_code(cache_key)
        if cached_result is not None:
            return cached_result
        result = func(*args, **kwargs)
        cache_code(result, cache_key)
        return result
    return wrapper

def cache_method(func):
    def wrapper(self, *args, **kwargs):
        cache_key = hashlib.md5(pickle.dumps((func.__name__, self, args, kwargs))).hexdigest()
        cached_result = load_cached_code(cache_key)
        if cached_result is not None:
            return cached_result
        result = func(self, *args, **kwargs)
        cache_code(result, cache_key)
        return result
    return wrapper

def clear_cache():
    for filename in os.listdir(CACHE_DIR):
        file_path = os.path.join(CACHE_DIR, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)

def cache_decorator(cache_key_func):
    def decorator(func):
        def wrapper(*args, **kwargs):
            cache_key = cache_key_func(func, *args, **kwargs)
            cached_result = load_cached_code(cache_key)
            if cached_result is not None:
                return cached_result
            result = func(*args, **kwargs)
            cache_code(result, cache_key)
            return result
        return wrapper
    return decorator

@cache_decorator(lambda func, *args, **kwargs: hashlib.md5(pickle.dumps((func.__name__, args, kwargs))).hexdigest())
def example_cached_function(a, b):
    return a * b

class CachedClass:
    def __init__(self, x):
        self.x = x
    
    @cache_method
    def add(self, y):
        return self.x + y
    
    @cache_method
    def multiply(self, y):
        return self.x * y
    
    @classmethod
    @cache_decorator(lambda func, x, y: hashlib.md5(pickle.dumps((func.__name__, x, y))).hexdigest())
    def example_class_method(cls, x, y):
        return x + y
    
    @staticmethod
    @cache_decorator(lambda func, x, y: hashlib.md5(pickle.dumps((func.__name__, x, y))).hexdigest())
    def example_static_method(x, y):
        return x * y
