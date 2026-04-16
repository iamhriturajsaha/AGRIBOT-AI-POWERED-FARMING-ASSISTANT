import os
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
import h5py
import json
MODEL_PATH = os.path.join(os.path.dirname(__file__), "disease_model.h5")
_model = None
def recursive_remove_keys(obj, keys_to_remove):
    changed = False
    if isinstance(obj, dict):
        for k in keys_to_remove:
            if k in obj:
                del obj[k]
                changed = True
        for k, v in list(obj.items()):
            if recursive_remove_keys(v, keys_to_remove):
                changed = True
    elif isinstance(obj, list):
        for item in obj:
            if recursive_remove_keys(item, keys_to_remove):
                changed = True
    return changed
def inject_missing_shapes(obj):
    changed = False
    if isinstance(obj, dict):
        if obj.get('class_name') == 'InputLayer' and 'config' in obj:
            layer_config = obj['config']
            if 'batch_shape' not in layer_config and 'shape' not in layer_config and 'batch_input_shape' not in layer_config:
                layer_config['batch_shape'] = [None, 128, 128, 3]
                changed = True
        for k, v in obj.items():
            if inject_missing_shapes(v):
                changed = True
    elif isinstance(obj, list):
        for item in obj:
            if inject_missing_shapes(item):
                changed = True
    return changed
def fix_keras3_incompatibilities(filepath):
    try:
        import zipfile
        import tempfile
        import shutil
        if zipfile.is_zipfile(filepath):
            with tempfile.TemporaryDirectory() as tmpdir:
                with zipfile.ZipFile(filepath, 'r') as z:
                    z.extractall(tmpdir)
                config_path = os.path.join(tmpdir, "config.json")
                if os.path.exists(config_path):
                    with open(config_path, "r", encoding="utf-8") as f:
                        config = json.load(f)
                    keys_to_strip = ['quantization_config']
                    changed_strip = recursive_remove_keys(config, keys_to_strip)
                    changed_inject = inject_missing_shapes(config)
                    if changed_strip or changed_inject:
                        with open(config_path, "w", encoding="utf-8") as f:
                            json.dump(config, f)
                        with zipfile.ZipFile(filepath, 'w', zipfile.ZIP_DEFLATED) as z:
                            for root, _, files in os.walk(tmpdir):
                                for file in files:
                                    f_path = os.path.join(root, file)
                                    arcname = os.path.relpath(f_path, tmpdir)
                                    arcname = arcname.replace(os.path.sep, '/')
                                    z.write(f_path, arcname)
                        print("✅ Successfully patched .keras config inside zip.")
        else:
            with h5py.File(filepath, 'r+') as f:
                if 'model_config' in f.attrs:
                    config_str = f.attrs['model_config']
                    if isinstance(config_str, bytes):
                        config_str = config_str.decode('utf-8')
                    config = json.loads(config_str)
                    keys_to_strip = ['quantization_config']
                    changed_strip = recursive_remove_keys(config, keys_to_strip)
                    changed_inject = inject_missing_shapes(config)
                    if changed_strip or changed_inject:
                        new_config_str = json.dumps(config)
                        f.attrs['model_config'] = new_config_str.encode('utf-8')
                        print("✅ Successfully restored batch_shape and stripped outdated kwargs in HDF5.")
    except Exception as e:
        import traceback
        print(f"Skipping patch step: {e}")
        traceback.print_exc()
def load_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
        fix_keras3_incompatibilities(MODEL_PATH)
        _model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        print("Model loaded from", MODEL_PATH)
    return _model

import numpy as np
import cv2

def make_gradcam_heatmap(img_array, model, pred_index=None):
    last_conv_layer_name = None
    target_model = model
    
    # Try finding it in the main model
    for layer in reversed(model.layers):
        try:
            shape = getattr(layer, 'output_shape', None)
            if isinstance(shape, tuple) and len(shape) == 4 and layer.name != 'sequential':
                last_conv_layer_name = layer.name
                break
        except Exception:
            pass
            
    # Check if it's a nested model (like MobileNetV2 inside Sequential)
    if not last_conv_layer_name:
        for layer in reversed(model.layers):
            if hasattr(layer, 'layers'):
                for inner_layer in reversed(layer.layers):
                    try:
                        shape = getattr(inner_layer, 'output_shape', None)
                        if isinstance(shape, tuple) and len(shape) == 4:
                            last_conv_layer_name = inner_layer.name
                            target_model = layer
                            break
                    except Exception:
                        pass
            if last_conv_layer_name:
                break
                
    if not last_conv_layer_name:
        return None
        
    last_conv_layer = target_model.get_layer(last_conv_layer_name)

    try:
        grad_model = tf.keras.models.Model(
            target_model.inputs, 
            [last_conv_layer.output, target_model.output]
        )

        with tf.GradientTape() as tape:
            # If the architecture wraps the image differently, this might need fallback
            # but usually it's fine.
            last_conv_layer_output, preds = grad_model(img_array)
            if pred_index is None:
                pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]

        grads = tape.gradient(class_channel, last_conv_layer_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        return heatmap.numpy()
    except Exception as e:
        print(f"Grad-CAM Error: {e}")
        return None

def save_and_display_gradcam(img_path, heatmap, cam_path, alpha=0.4):
    if heatmap is None:
        return None
    img = cv2.imread(img_path)
    if img is None:
        return None
        
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = heatmap * alpha + img
    cv2.imwrite(cam_path, superimposed_img)
    return cam_path