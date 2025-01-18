from src.model.train import TrainModel

if __name__ == "__main__":
    try:
        TrainModel().train_and_log_model()
    except Exception as e:  
        print(f"Error in training model: {e}")
        raise
    