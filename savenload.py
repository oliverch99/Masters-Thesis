import pickle
import functools
import inspect
import os


def safe_filename(s):
    return "".join([c if c.isalnum() or c in [' ', '.', '_'] else '_' for c in s])

def save_and_load_data(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Generate a base filename based on the function name
        base_filename = f"{func.__name__}_"
        # Serialize args and kwargs into strings, replace problematic characters
        args_str = '_'.join(safe_filename(str(arg)) for arg in args)
        kwargs_str = '_'.join(f"{k}{safe_filename(str(v))}" for k, v in kwargs.items())

        # Combine everything into a final filename
        filename = f"{base_filename}{args_str}_{kwargs_str}.pkl"
        filepath = os.path.join("data", filename)

        # Ensure the 'alpaca' directory exists
        os.makedirs("data", exist_ok=True)

        # Check if data for these arguments already exists
        if os.path.exists(filepath):
            print(f"Loading saved data for {func.__name__} with args {args} and kwargs {kwargs}")
            with open(filepath, "rb") as file:
                return pickle.load(file)

        # If not, execute the function and save the result
        result = func(*args, **kwargs)
        with open(filepath, "wb") as file:
            pickle.dump(result, file)
        
        return result
    
    return wrapper


def Save_and_load_data(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        # Generate a base filename based on the function name
        base_filename = f"{func.__name__}_"
        # Serialize args and kwargs into strings, replace problematic characters
        args_str = '_'.join(safe_filename(str(arg)) for arg in args)
        kwargs_str = '_'.join(f"{k}{safe_filename(str(v))}" for k, v in kwargs.items())

        # Combine everything into a final filename
        filename = f"{base_filename}{args_str}_{kwargs_str}.pkl"
        filepath = os.path.join("data", filename)

        # Ensure the 'alpaca' directory exists
        os.makedirs("data", exist_ok=True)

        # Check if data for these arguments already exists
        if os.path.exists(filepath):
            print(f"Loading saved data for {func.__name__} with args {args} and kwargs {kwargs}")
            with open(filepath, "rb") as file:
                return pickle.load(file)

        # If not, execute the function and save the result
        result = func(self, *args, **kwargs)
        with open(filepath, "wb") as file:
            pickle.dump(result, file)
        
        return result
    
    return wrapper
