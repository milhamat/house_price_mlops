import mlflow
import psutil
import time

# Define a function to log system metrics
def log_system_metrics(interval=1):
    """
    Logs system metrics (CPU, memory, and disk) at regular intervals.
    :param interval: Time in seconds between logging.
    """
    try:
        while True:
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
    except KeyboardInterrupt:
        print("Stopped logging system metrics.")

# # Start an MLflow run
# with mlflow.start_run():
#     # Start logging system metrics in a separate thread or process
#     log_system_metrics(interval=5)  # Log every 5 seconds
