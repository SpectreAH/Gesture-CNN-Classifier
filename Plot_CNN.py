import os
import torch
from torchview import draw_graph

# Assuming your models are in a file named 'models.py'
from models import build_model  

# 1. Ensure the target directory exists
output_dir = "runs/small/eval"
os.makedirs(output_dir, exist_ok=True)

# 2. Build your main HandGestureCNN model using your factory function
# Default configuration: 1 channel, 128x128 images, 10 classes
model = build_model(name="small", num_classes=10)

# 3. Create the execution graph
# HandGestureCNN expects a batch of images: (batch_size, channels, height, width)
model_graph = draw_graph(
    model, 
    input_size=(1, 1, 128, 128), 
    expand_nested=True,          # Unrolls your ConvBlock layers to show inside them
    depth=3,                     # Controls how deep into nested blocks to display
    device="cpu"
)

# 4. Save the plot into your evaluation folder
output_path = os.path.join(output_dir, "hand_gesture_cnn_small_batch_architecture")
model_graph.visual_graph.render(
    filename=output_path, 
    format="png",
    cleanup=True  # Removes temporary Graphviz source files, leaving only the PNG
)

print(f"Success! CNN architecture plot saved to: {output_path}.png")