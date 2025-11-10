import os
import json
import base64
import io
import numpy as np
from PIL import Image

# Example: import your DEXTR inference library; adjust as needed
import cv2

def init_context(context):
    context.logger.info('Initializing DEXTR model...')
    # Path to your model file inside the container
    model_path = os.environ.get('MODEL_PATH', '/opt/nuclio/model/dextr.onnx')
    # Example: load ONNX model (or whatever format you have)
    import onnxruntime as rt
    sess = rt.InferenceSession(model_path)
    context.user_data.sess = sess
    context.user_data.input_name = sess.get_inputs()[0].name
    context.user_data.output_name = sess.get_outputs()[0].name
    context.logger.info('Model loaded from %s', model_path)

def handler(context, event):
    try:
        body = event.body
        # Expecting an image in base64
        img_b64 = body.get('image')
        if img_b64 is None:
            raise ValueError('No image provided in request')

        img_bytes = base64.b64decode(img_b64)
        image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        img_np = np.array(image)

        # Pre-process for DEXTR (example): resize / normalize etc
        # Adjust according to your model’s expected input
        input_blob = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        input_blob = cv2.resize(input_blob, (256, 256))
        input_blob = input_blob.transpose(2, 0, 1).astype(np.float32) / 255.0
        input_blob = np.expand_dims(input_blob, axis=0)

        sess = context.user_data.sess
        result = sess.run([context.user_data.output_name],
                          {context.user_data.input_name: input_blob})[0]

        # Post-process: convert output mask to polygon or segmentation result
        # This is just an example placeholder
        mask = (result[0,0] > 0.5).astype(np.uint8) * 255
        # Optionally find contours and output polygon points
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polygons = []
        for cnt in contours:
            pts = cnt.squeeze().tolist()
            polygons.append(pts)

        response = {
            'polygons': polygons,
            'type': 'polygon'
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
