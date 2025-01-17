import os

def get_artifact_path(*path_parts):
    """
    Helper function to construct a full path to an artifact file.
    
    Args:
        path_parts: List of path components relative to the `artifacts/` directory.
        
    Returns:
        str: Full path to the artifact.
    """
    return os.path.join("artifacts", *path_parts)


def log_message(message, level="INFO"):
    """
    Helper function to log messages to the console.
    
    Args:
        message (str): The message to log.
        level (str): The log level (default is "INFO").
    """
    print(f"[{level}] {message}")
