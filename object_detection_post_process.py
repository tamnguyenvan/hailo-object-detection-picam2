import cv2
import numpy as np
from toolbox import id_to_color


def inference_result_handler(original_frame, infer_results, labels, config_data, tracker=None, draw_trail=False):
    """
    Processes inference results and draw detections (with optional tracking).

    Args:
        infer_results (list): Raw output from the model.
        original_frame (np.ndarray): Original image frame.
        labels (list): List of class labels.
        enable_tracking (bool): Whether tracking is enabled.
        tracker (BYTETracker, optional): ByteTrack tracker instance.

    Returns:
        np.ndarray: Frame with detections or tracks drawn.
    """
    detections = extract_detections(original_frame, infer_results, config_data)  # Should return dict with boxes, classes, scores
    frame_with_detections = draw_detections(detections, original_frame, labels, tracker=tracker)
    return frame_with_detections

def draw_detection(image: np.ndarray, box: list, labels: list, score: float, color: tuple, track=False):
    """
    Draw premium box and semi-transparent label for one detection.
    Optimized: only blends the ROI to maintain high FPS.
    """
    xmin, ymin, xmax, ymax = map(int, box)
    
    # Bounding box
    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2, cv2.LINE_AA)
    
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 0.5
    thickness = 1
    
    # Compose text
    text = f"{labels[0]} {score:.0f}%"
    if track and len(labels) > 1:
        text = f"{labels[0]} {labels[1]}"
        if len(labels) > 2:
            text += f" | {labels[2]}"

    # Get text size
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    # Define label region
    label_ymin = max(ymin - text_h - 10, 0)
    label_ymax = ymin
    label_xmin = xmin
    label_xmax = xmin + text_w + 10
    
    # Ensure label region is within image bounds
    img_h, img_w = image.shape[:2]
    label_xmax = min(label_xmax, img_w)
    label_ymax = min(label_ymax, img_h)

    # ROI-based Alpha Blending
    if label_ymax > label_ymin and label_xmax > label_xmin:
        roi = image[label_ymin:label_ymax, label_xmin:label_xmax]
        overlay = roi.copy()
        cv2.rectangle(overlay, (0, 0), (label_xmax - label_xmin, label_ymax - label_ymin), color, -1)
        alpha = 0.6
        cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0, roi)
    
    # Draw text on top
    text_y = ymin - 7 if ymin - 7 > 7 else ymin + text_h + 7
    cv2.putText(image, text, (xmin + 5, text_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)


def denormalize_and_rm_pad(box: list, size: int, padding_length: int, input_height: int, input_width: int) -> list:
    """
    Denormalize bounding box coordinates and remove padding.

    Args:
        box (list): Normalized bounding box coordinates.
        size (int): Size to scale the coordinates.
        padding_length (int): Length of padding to remove.
        input_height (int): Height of the input image.
        input_width (int): Width of the input image.

    Returns:
        list: Denormalized bounding box coordinates with padding removed.
    """
    # Scale box coordinates
    box = [int(x * size) for x in box]

    # Apply padding correction
    for i in range(4):
        if i % 2 == 0:  # x-coordinates
            if input_height != size:
                box[i] -= padding_length
        else:  # y-coordinates
            if input_width != size:
                box[i] -= padding_length

    # Swap to [ymin, xmin, ymax, xmax]
    return [box[1], box[0], box[3], box[2]]


def extract_detections(image: np.ndarray, detections: list, config_data) -> dict:
    """
    Extract detections from the input data.

    Args:
        image (np.ndarray): Image to draw on.
        detections (list): Raw detections from the model.
        config_data (Dict): Loaded JSON config containing post-processing metadata.

    Returns:
        dict: Filtered detection results containing 'detection_boxes', 'detection_classes', 'detection_scores', and 'num_detections'.
    """

    visualization_params = config_data["visualization_params"]
    score_threshold = visualization_params.get("score_thres", 0.5)
    max_boxes = visualization_params.get("max_boxes_to_draw", 50)

    img_height, img_width = image.shape[:2]
    size = max(img_height, img_width)
    padding_length = int(abs(img_height - img_width) / 2)

    all_detections = []

    for class_id, detection in enumerate(detections):
        for det in detection:
            bbox, score = det[:4], det[4]
            if score >= score_threshold:
                denorm_bbox = denormalize_and_rm_pad(bbox, size, padding_length, img_height, img_width)
                all_detections.append((score, class_id, denorm_bbox))

    # Sort all detections by score descending
    all_detections.sort(reverse=True, key=lambda x: x[0])

    # Take top max_boxes
    top_detections = all_detections[:max_boxes]

    scores, class_ids, boxes = zip(*top_detections) if top_detections else ([], [], [])

    return {
        'detection_boxes': list(boxes),
        'detection_classes': list(class_ids),
        'detection_scores': list(scores),
        'num_detections': len(top_detections)
    }


def draw_detections(detections: dict, img_out: np.ndarray, labels, tracker=None, track_start_times=None) -> np.ndarray:
    """
    Draw detections or tracking results with object timers and refined aesthetics.

    Args:
        detections (dict): Raw detection outputs.
        img_out (np.ndarray): Image to draw on.
        labels (list): List of class labels.
        tracker (BYTETracker, optional): ByteTrack tracker instance.
        track_start_times (dict, optional): Dict mapping track_id to start time.

    Returns:
        np.ndarray: Annotated image.
    """
    import time

    # Extract detection data
    boxes = detections["detection_boxes"]
    scores = detections["detection_scores"]
    num_detections = detections["num_detections"]
    classes = detections["detection_classes"]

    if tracker:
        dets_for_tracker = []
        for idx in range(num_detections):
            box = boxes[idx]
            score = scores[idx]
            dets_for_tracker.append([*box, score])

        # Logic for "someone blocking the camera" is handled in demo.py by skipping this update
        if not dets_for_tracker:
            return img_out

        online_targets = tracker.update(np.array(dets_for_tracker))

        for track in online_targets:
            track_id = track.track_id
            x1, y1, x2, y2 = track.tlbr
            xmin, ymin, xmax, ymax = map(int, [x1, y1, x2, y2])
            best_idx = find_best_matching_detection_index(track.tlbr, boxes)
            
            # Default color from class
            color = (255, 255, 255) # Fallback White
            class_label = "Unknown"
            
            if best_idx is not None:
                class_id = classes[best_idx]
                color = tuple(id_to_color(class_id).tolist())
                if class_id < len(labels):
                    class_label = labels[class_id]
                else:
                    class_label = f"Class {class_id}"

            # Timer Logic
            display_labels = [class_label, f"ID {track_id}"]
            if track_start_times is not None:
                if track_id not in track_start_times:
                    track_start_times[track_id] = time.time()
                
                elapsed = time.time() - track_start_times[track_id]
                mins = int(elapsed // 60)
                secs = int(elapsed % 60)
                time_str = f"{mins}:{secs:02d}"
                display_labels.append(time_str)

                # Turn red if > 60 seconds
                if elapsed > 60:
                    color = (255, 0, 0) # RGB Red

            draw_detection(img_out, [xmin, ymin, xmax, ymax], display_labels,
                           track.score * 100.0, color, track=True)



    else:
        # No tracking — draw raw model detections
        for idx in range(num_detections):
            color = tuple(id_to_color(classes[idx]).tolist())  # Color based on class
            draw_detection(img_out, boxes[idx], [labels[classes[idx]]], scores[idx] * 100.0, color)

    return img_out


def find_best_matching_detection_index(track_box, detection_boxes):
    """
    Finds the index of the detection box with the highest IoU relative to the given tracking box.

    Args:
        track_box (list or tuple): The tracking box in [x_min, y_min, x_max, y_max] format.
        detection_boxes (list): List of detection boxes in [x_min, y_min, x_max, y_max] format.

    Returns:
        int or None: Index of the best matching detection, or None if no match is found.
    """
    best_iou = 0
    best_idx = -1

    for i, det_box in enumerate(detection_boxes):
        iou = compute_iou(track_box, det_box)
        if iou > best_iou:
            best_iou = iou
            best_idx = i

    return best_idx if best_idx != -1 else None


def compute_iou(boxA, boxB):
    """
    Compute Intersection over Union (IoU) between two bounding boxes.

    IoU measures the overlap between two boxes:
        IoU = (area of intersection) / (area of union)
    Values range from 0 (no overlap) to 1 (perfect overlap).

    Args:
        boxA (list or tuple): [x_min, y_min, x_max, y_max]
        boxB (list or tuple): [x_min, y_min, x_max, y_max]

    Returns:
        float: IoU value between 0 and 1.
    """
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = max(1e-5, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    areaB = max(1e-5, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
    return inter / (areaA + areaB - inter + 1e-5)
