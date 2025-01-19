import mlflow
import psutil
import time

# Define a function to log system metrics
def log_system_metrics(interval: int = 1, max_iterations: int = 5) -> None:
    """
    Logs system metrics (CPU, memory, and disk) for a specified number of iterations.
    :param interval: Time in seconds between logging.
    :param max_iterations: Number of times to log metrics before stopping.
    """
    try:
        for _ in range(max_iterations):
            # Capture system metrics
            cpu_usage = psutil.cpu_percent(interval=None)
            memory_usage = psutil.virtual_memory().percent
            disk_usage = psutil.disk_usage('/').percent

            # Log metrics to MLflow
            mlflow.log_metric("cpu_usage", cpu_usage)
            mlflow.log_metric("memory_usage", memory_usage)
            mlflow.log_metric("disk_usage", disk_usage)

            print(f"Logged CPU: {cpu_usage}%, Memory: {memory_usage}%, Disk: {disk_usage}%")
            time.sleep(interval)

        print("Completed logging system metrics.")
    except Exception as e:
        print(f"An error occurred: {e}")

