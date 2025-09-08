import subprocess
import sys

def run_classification_script(model_size, architecture):
    """
    Runs the train_classifier.py script as a subprocess and returns its output.
    """
    command = [
        sys.executable,  # Use the same python interpreter that's running flask
        'train_classifier.py',
        '--model-size',
        model_size
    ]

    print(f"--- Running Subprocess ---")
    print(f"Command: {' '.join(command)}")

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,  # This will raise a CalledProcessError if the script returns a non-zero exit code
            timeout=300 # 5 minute timeout for the training script
        )
        print("--- Subprocess Finished Successfully ---")
        # Return the standard output of the script
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"--- Subprocess Failed ---")
        # Return the standard error if the script fails
        error_message = f"Training script failed with exit code {e.returncode}.\n\n--- STDERR ---\n{e.stderr}"
        return error_message
    except subprocess.TimeoutExpired as e:
        print(f"--- Subprocess Timed Out ---")
        return f"Training script timed out after {e.timeout} seconds."
    except FileNotFoundError:
        return "Error: 'train_classifier.py' not found. Make sure the script is in the correct directory."


def run_segmentation_script(model_size, architecture):
    """
    Runs the train_segmentation.py script as a subprocess and returns its output.
    """
    command = [
        sys.executable,
        'train_segmentation.py',
        '--model-size',
        model_size
    ]

    print(f"--- Running Subprocess ---")
    print(f"Command: {' '.join(command)}")

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
            timeout=300
        )
        print("--- Subprocess Finished Successfully ---")
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"--- Subprocess Failed ---")
        return f"Training script failed with exit code {e.returncode}.\n\n--- STDERR ---\n{e.stderr}"
    except subprocess.TimeoutExpired as e:
        print(f"--- Subprocess Timed Out ---")
        return f"Training script timed out after {e.timeout} seconds."
    except FileNotFoundError:
        return "Error: 'train_segmentation.py' not found."

def run_training_job(task_type, model_size, architecture):
    """
    Dispatcher function to run the correct training job based on task_type.
    """
    if task_type == "Image Classification":
        return run_classification_script(model_size, architecture)
    elif task_type in ["Semantic Segmentation", "Instance Segmentation"]:
        return run_segmentation_script(model_size, architecture)
    # --- Other tasks still use placeholders for now ---
    elif task_type == "Object Detection":
        print(f"--- Starting placeholder training for {task_type} ---")
        return f"Placeholder: Training for {task_type} with {architecture} and {model_size} would start here."
    elif task_type == "Anomaly Detection":
        print(f"--- Starting placeholder training for {task_type} ---")
        return f"Placeholder: Training for {task_type} with {architecture} and {model_size} would start here."
    elif task_type == "Depth Estimation":
        print(f"--- Starting placeholder training for {task_type} ---")
        return f"Placeholder: Training for {task_type} with {architecture} and {model_size} would start here."
    else:
        return f"Error: No training logic defined for task type '{task_type}'."
