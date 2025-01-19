import os
# import yaml

class Config:
    """
    Class to store configuration parameters for the project.
    """
    # # Define the path to the data directory
    # DATA_DIR = "data"
    # # Define the path to the artifacts directory
    # ARTIFACTS_DIR = "artifacts"

    def get_artifact_path(*path_parts: str) -> str:
        """
        Helper function to construct a full path to an artifact file.
        
        Args:
            path_parts: List of path components relative to the `artifacts/` directory.
            
        Returns:
            str: Full path to the artifact.
        """
        return os.path.join("artifacts", *path_parts)


