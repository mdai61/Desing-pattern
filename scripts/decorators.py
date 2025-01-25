def log_results(func):
    """Decorator to log results of a function."""
    def wrapper(*args, **kwargs):
        print(f"Running: {func.__name__}")
        result = func(*args, **kwargs)
        if isinstance(result, dict):
            print("Results:", {key: round(value, 4) if isinstance(value, float) else value for key, value in result.items()})
        return result
    return wrapper
