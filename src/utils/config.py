import os
# import yaml

class Config:
    """
    Class to store configuration parameters for the project.
    """
    # Define the path to the data directory
    DATA_DIR = "data"
    # Define the path to the artifacts directory
    ARTIFACTS_DIR = "artifacts"

    def get_artifact_path(*path_parts: str) -> str:
        """
        Helper function to construct a full path to an artifact file.
        
        Args:
            path_parts: List of path components relative to the `artifacts/` directory.
            
        Returns:
            str: Full path to the artifact.
        """
        return os.path.join("artifacts", *path_parts)


    def log_message(message:str, level="INFO"):
        """
        Helper function to log messages to the console.
        
        Args:
            message (str): The message to log.
            level (str): The log level (default is "INFO").
        """
        print(f"[{level}] {message}")

# def read_yaml(file_path: str)-> dict:
#     """ Function to read yaml file

#     Args:
#         file_path (_type_): The path to the yaml file

#     Returns:
#         _type_: The content of the yaml file
#     """
#     with open(file_path, 'r') as stream:
#         try:
#             return yaml.safe_load(stream)
#         except yaml.YAMLError as exc:
#             print(exc)
#             return None