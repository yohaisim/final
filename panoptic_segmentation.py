import torch
from PIL import Image
import numpy as np
from transformers import AutoImageProcessor, AutoModelForUniversalSegmentation

def panoptic_segmentation(image_path):

    # Load original image
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)

    orig_width, orig_height = image.width, image.height

    # Load pretrained Mask2Former model (COCO panoptic)
    processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-large-coco-panoptic")
    model = AutoModelForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-large-coco-panoptic")

    # Prepare image for the model
    inputs = processor(images=image, return_tensors="pt")

    # Run inference
    with torch.no_grad():
        outputs = model(**inputs)

    # Postprocess the segmentation output
    results = processor.post_process_panoptic_segmentation(
        outputs=outputs,
        threshold=0.60,
        mask_threshold=0.80,
        target_sizes=[(orig_height, orig_width)]
    )

    panoptic_result = results[0]
    panoptic_mask = panoptic_result["segmentation"].cpu().numpy()
    segments_info = panoptic_result["segments_info"]

    # Assign a unique color per segment
    h, w = panoptic_mask.shape
    seg_colors = np.zeros((h, w, 3), dtype=np.uint8)
    rnd = np.random.RandomState(42)

    segments_data = []
    for seg_info in segments_info:
        seg_id = seg_info["id"]
        label_id = seg_info["label_id"]
        score = seg_info["score"]
        class_name = model.config.id2label[label_id]

        mask = (panoptic_mask == seg_id)
        color = rnd.randint(0, 256, size=3, dtype=np.uint8)
        seg_colors[mask] = color

        segments_data.append({
            "seg_id": seg_id,
            "class_name": class_name,
            "score": float(score),
        })

    return seg_colors, segments_data, panoptic_mask
