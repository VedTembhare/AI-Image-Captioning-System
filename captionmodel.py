from ultralytics import YOLO
from transformers import BlipProcessor, BlipForConditionalGeneration
import pandas as pd
import numpy as np
import cv2

# Load models
yolo_model = YOLO("yolov8n.pt")

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")


def generate_heatmap(image, results):
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    heatmap = np.zeros(img.shape[:2], dtype=np.float32)

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        heatmap[y1:y2, x1:x2] += conf

    heatmap = np.clip(heatmap, 0, 1)
    heatmap = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    return overlay


def enhance_caption(base_caption, objects, confidences):
    strong_objects = [obj for obj, conf in zip(objects, confidences) if conf > 0.5]
    unique_objects = list(set(strong_objects))

    if len(unique_objects) > 0:
        obj_text = ", ".join(unique_objects[:4])
        return f"{base_caption}. The scene clearly contains {obj_text}."

    return base_caption


def get_confidence_label(conf):
    """Return human-readable confidence tier."""
    if conf >= 0.85:
        return "very high", "🟢"
    elif conf >= 0.70:
        return "high", "🟡"
    elif conf >= 0.50:
        return "moderate", "🟠"
    else:
        return "low", "🔴"


def get_size_description(x1, y1, x2, y2, img_w, img_h):
    """Describe object size relative to the image."""
    obj_area = (x2 - x1) * (y2 - y1)
    img_area = img_w * img_h
    ratio = obj_area / img_area

    if ratio > 0.4:
        return "large (dominates the frame)"
    elif ratio > 0.15:
        return "medium-sized"
    elif ratio > 0.04:
        return "small"
    else:
        return "very small"


def get_position_description(x1, y1, x2, y2, img_w, img_h):
    """Describe where in the image the object is located."""
    cx = (x1 + x2) / 2 / img_w
    cy = (y1 + y2) / 2 / img_h

    v = "top" if cy < 0.33 else ("bottom" if cy > 0.66 else "middle")
    h = "left" if cx < 0.33 else ("right" if cx > 0.66 else "center")

    if h == "center" and v == "middle":
        return "at the center of the image"
    return f"in the {v}-{h} region"


def explain_detections(image, results):
    """
    Generate natural-language XAI explanations for each detected object.

    Returns a list of dicts with keys:
        object, confidence, conf_label, conf_icon,
        size, position, explanation, crop
    """
    img_array = np.array(image)
    img_h, img_w = img_array.shape[:2]

    explanations = []

    for box in results[0].boxes:
        cls = int(box.cls[0])
        label = yolo_model.names[cls]
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Clamp to image bounds
        x1c, y1c = max(0, x1), max(0, y1)
        x2c, y2c = min(img_w, x2), min(img_h, y2)

        conf_label, conf_icon = get_confidence_label(conf)
        size_desc = get_size_description(x1, y1, x2, y2, img_w, img_h)
        position_desc = get_position_description(x1, y1, x2, y2, img_w, img_h)

        # Crop the detected region for display
        crop = img_array[y1c:y2c, x1c:x2c]

        # Build natural language explanation
        explanation = (
            f"**Why '{label}' was detected:**\n"
            f"The model identified a **{label}** {position_desc} with **{conf_label} confidence ({conf:.1%})**. "
            f"The detected region is {size_desc}. "
            f"YOLOv8 recognized the shape, texture, and spatial features of this region as strongly matching "
            f"the learned visual pattern for '{label}' in its training data. "
        )

        if conf >= 0.85:
            explanation += (
                f"The very high confidence score suggests the visual features — such as edges, proportions, "
                f"and context — are a near-perfect match for '{label}'."
            )
        elif conf >= 0.70:
            explanation += (
                f"The high confidence indicates strong agreement between the detected region's features "
                f"and the model's internal representation of '{label}'."
            )
        elif conf >= 0.50:
            explanation += (
                f"The moderate confidence means the region partially matches '{label}', "
                f"but some visual ambiguity exists — possibly due to occlusion, unusual angle, or lighting."
            )
        else:
            explanation += (
                f"The low confidence suggests this may be a partial or unclear match. "
                f"The object might be occluded, at an unusual angle, or similar to another class."
            )

        explanations.append({
            "object": label,
            "confidence": conf,
            "conf_label": conf_label,
            "conf_icon": conf_icon,
            "size": size_desc,
            "position": position_desc,
            "explanation": explanation,
            "crop": crop if crop.size > 0 else None
        })

    # Sort by confidence descending
    explanations.sort(key=lambda x: x["confidence"], reverse=True)
    return explanations


def analyzeimage(image):
    results = yolo_model(image)

    labels = []
    confidences = []

    for box in results[0].boxes:
        cls = int(box.cls[0])
        label = yolo_model.names[cls]
        conf = float(box.conf[0])
        labels.append(label)
        confidences.append(conf)

    df = pd.DataFrame({
        "Object": labels,
        "Confidence": confidences
    })

    annotatedimg = results[0].plot()

    # BLIP caption
    inputs = processor(images=image, return_tensors="pt")
    out = blip_model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)

    # Enhanced caption
    caption = enhance_caption(caption, labels, confidences)

    heatmap = generate_heatmap(image, results)

    # XAI explanations
    xai_explanations = explain_detections(image, results)

    return caption, annotatedimg, df, heatmap, xai_explanations