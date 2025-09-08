from flask import Flask, render_template, jsonify, request
from training_logic import run_training_job

app = Flask(__name__)

# Data from dev_doc.md
TASK_TO_ARCHITECTURES = {
    "Image Classification": ["DINOv3Classifier"],
    "Semantic Segmentation": ["DINOv3SegmentationHead"],
    "Instance Segmentation": ["MaskDINO"],
    "Object Detection": ["DINOv3FasterRCNN", "DETR"],
    "Anomaly Detection": ["AnomalyDINO"],
    "Depth Estimation": ["DINOv3DepthEstimator"],
}

DINOV3_MODELS = [
    "ViT-S/16 (21M)",
    "ViT-B/16",
    "ViT-L/16",
    "ViT-7B/16 (7B)",
    "ConvNeXt-T",
    "ConvNeXt-S",
    "ConvNeXt-B",
    "ConvNeXt-L",
]

@app.route('/')
def index():
    """Renders the main page with initial data."""
    task_types = list(TASK_TO_ARCHITECTURES.keys())
    # Pass the initial list of architectures for the default task
    initial_architectures = TASK_TO_ARCHITECTURES[task_types[0]]
    return render_template(
        'index.html',
        task_types=task_types,
        initial_architectures=initial_architectures,
        model_sizes=DINOV3_MODELS
    )

@app.route('/api/architectures/<task_type>')
def get_architectures(task_type):
    """Returns a list of model architectures for a given task type."""
    architectures = TASK_TO_ARCHITECTURES.get(task_type, [])
    return jsonify(architectures)

@app.route('/train', methods=['POST'])
def train():
    """Handles the training request from the frontend."""
    data = request.get_json()
    task_type = data.get('task_type')
    model_arch = data.get('model_arch')
    model_size = data.get('model_size')

    if not all([task_type, model_arch, model_size]):
        return jsonify({'status': 'Error', 'message': 'Missing required parameters.'}), 400

    print(f"Received training request: Task={task_type}, Arch={model_arch}, Size={model_size}")

    # Call the dispatcher function from the training logic module
    result_message = run_training_job(task_type, model_size, model_arch)

    return jsonify({'status': 'Success', 'message': result_message})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
