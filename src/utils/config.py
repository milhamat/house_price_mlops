import os
# import yaml

class Config:
    PROJECTNAME = "House Price Prediction v.1"
    
    DATA_NAME = "train.csv"
    
    ARTIFACTS_DIR = "artifacts"
    
    ARTIFACT_DATASET = "datasets"
    
    ARTIFACT_MODEL = "models"
    
    MODEL_NAME = "model.pkl"
    
    

def get_artifact_path(*path_parts: str) -> str:
    """
    Helper function to construct a full path to an artifact file.
        
    Args:
        path_parts: List of path components relative to the `artifacts/` directory.
            
    Returns:
        str: Full path to the artifact.
    """
    return os.path.join("artifacts", *path_parts)


