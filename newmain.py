import os
import json
import base64
import io
import numpy as np
from PIL import Image
import cv2
import onnxruntime as rt

# --- Constants ---
# The (Width, Height) your ONNX model expects
# Used for resizing the image and points
NET_INPUT_SIZE = (256, 256) 

def init_context(context):
    """
    Initializes the ONNX model session and finds its input/output names.
    """
    context.logger.info('Initializing DEXTR model...')
    model_path = os.environ.get('MODEL_PATH', '/opt/nuclio/model/dextr.onnx')
    
    sess = rt.InferenceSession(model_path)
    context.user_data.sess = sess
    
    inputs = sess.get_inputs()
    if len(inputs) != 2:
        context.logger.error(f"Model expects 2 inputs (image, points), but found {len(inputs)}")
        raise ValueError("Invalid model: DEXTR requires 2 inputs")

    # DEXTR models have two inputs: one for the image, one for the points.
    # The order might vary, but typically:
    # inputs[0] is the image (e.g., shape [1, 3, 256, 256])
    # inputs[1] is the points heatmap (e.g., shape [1, 1, 256, 256])
    context.user_data.input_name_image = inputs[0].name
    context.user_data.input_name_points = inputs[1].name
    
    context.user_data.output_name = sess.get_outputs()[0].name
    context.user_data.net_size = NET_INPUT_SIZE # (Width, Height)

    context.logger.info(f"Model loaded: {model_path}")
    context.logger.info(f"Input Image: {context.user_data.input_name_image}")
    context.logger.info(f"Input Points: {context.user_data.input_name_points}")


def _preprocess_points(points, original_shape, net_size):
    """
    Converts CVAT's [x, y] points into a heatmap blob for the DEXTR model.
    
    :param points: List of [x, y] coordinates from CVAT.
    :param original_shape: (Height, Width) of the original image.
    :param net_size: (Width, Height) of the model's input.
    :return: A numpy blob (heatmap) for the model's 'points' input.
    """
    original_h, original_w = original_shape
    net_w, net_h = net_size
    
    # Create an empty heatmap
    # Shape is (batch, channels, height, width)
    points_blob = np.zeros((1, 1, net_h, net_w), dtype=np.float32)
    
    for p in points:
        # CVAT sends points as [x, y]
        x, y = p
        
        # Scale coordinates from original size to network size
        scaled_x = int(x * (net_w / original_w))
        scaled_y = int(y * (net_h / original_h))
        
        # Clamp values to be within the network's dimensions
        scaled_x = max(0, min(scaled_x, net_w - 1))
        scaled_y = max(0, min(scaled_y, net_h - 1))
        
        # Set the corresponding pixel in the heatmap to 1.0
        points_blob[0, 0, scaled_y, scaled_x] = 1.0
            
    return points_blob


def handler(context, event):
    """
    Handles the inference request from CVAT.
    """
    try:
        body = event.body
        
        # 1. Get Image
        img_b64 = body.get('image')
        if img_b64 is None:
            raise ValueError('No image provided in request')

        img_bytes = base64.b64decode(img_b64)
        image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        img_np = np.array(image)
        
        # Store original image size (Height, Width)
        original_shape = img_np.shape[:2]
        
        # 2. Get Points (This is the critical missing piece)
        # CVAT sends interactor points in the 'state' object
        state = body.get('state')
        if not state:
            raise ValueError("Missing 'state' object for interactor")
            
        # DEXTR uses 'positive' points
        pos_points = state.get('pos_points')
        if not pos_points:
            raise ValueError("Missing 'pos_points' in state (at least 4 are required for DEXTR)")

        # 3. Pre-process Image
        net_w, net_h = context.user_data.net_size
        img_blob = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        img_blob = cv2.resize(img_blob, (net_w, net_h))
        img_blob = img_blob.transpose(2, 0, 1).astype(np.float32) / 255.0
        img_blob = np.expand_dims(img_blob, axis=0) # (1, 3, H, W)

        # 4. Pre-process Points
        points_blob = _preprocess_points(pos_points, original_shape, (net_w, net_h))

        # 5. Run Inference
        sess = context.user_data.sess
        result = sess.run(
            [context.user_data.output_name],
            {
                context.user_data.input_name_image: img_blob,
                context.user_data.input_name_points: points_blob
            }
        )[0]

        # 6. Post-process
        # The result is a (1, 1, net_h, net_w) mask
        mask = (result[0, 0] > 0.5).astype(np.uint8) * 255
        
        # **CRITICAL**: Resize the mask back to the *original* image size
        original_size_wh = (original_shape[1], original_shape[0]) # (W, H) for cv2.resize
        mask = cv2.resize(mask, original_size_wh, interpolation=cv2.INTER_NEAREST)

        # Find contours on the full-sized mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        polygons = []
        for cnt in contours:
            if len(cnt) < 3: # A polygon needs at least 3 points
                continue
            # .squeeze() removes redundant dimensions. .tolist() makes it JSON-serializable.
            pts = cnt.squeeze().tolist()
            if not isinstance(pts[0], list): # Handle single-point contours
                pts = [pts]
            polygons.append(pts)

        # 7. Format Response
        # CVAT interactors expect a 'mask' or 'polygons'. Polygons are more editable.
        response = {
            'polygons': polygons
            # Note: The 'type' key from your original code is not used by CVAT interactors.
        }
        return context.Response(body=json.dumps(response),
                                headers={},
                                content_type='application/json',
                                status_code=200)
                                
    except Exception as e:
        context.logger.error('Error during inference: %s', str(e))
        return context.Response(body=json.dumps({'error': str(e)}),
                                headers={},
                                content_type='application/json',
                                status_code=500)
