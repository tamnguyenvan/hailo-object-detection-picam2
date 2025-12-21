import argparse
import cv2
import numpy as np
import sys
import os
import threading
import queue
from loguru import logger
from types import SimpleNamespace

# Add current directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

try:
    import picamera2
except ImportError:
    logger.error("Picamera2 not found. Please run on a Raspberry Pi with Picamera2 installed.")
    sys.exit(1)

from hailo_inference import HailoInfer
from object_detection_post_process import extract_detections, draw_detections
from tracker.byte_tracker import BYTETracker
from toolbox import (
    get_labels,
    load_json_file,
    default_preprocess,
    FrameRateTracker,
)

def capture_thread_func(picam, model_w, model_h, input_queue, stop_event):
    logger.info("Capture thread started.")
    while not stop_event.is_set():
        frame = picam.capture_array()
        if frame is None:
            break
        
        if frame.shape[2] == 4:
            frame = frame[:, :, :3].copy()
        
        preprocessed_frame = default_preprocess(frame, model_w, model_h)
        
        try:
            # Use a small timeout to check stop_event periodically
            input_queue.put((frame, preprocessed_frame), timeout=1)
        except queue.Full:
            continue
    logger.info("Capture thread exiting.")

def inference_thread_func(hailo_inference, input_queue, output_queue, stop_event):
    logger.info("Inference thread started.")
    
    while not stop_event.is_set():
        try:
            item = input_queue.get(timeout=1)
        except queue.Empty:
            continue
            
        frame, preprocessed_frame = item
        inference_results = None
        
        # We need to capture the results in the callback
        def callback(completion_info, bindings_list):
            nonlocal inference_results
            if completion_info.exception:
                logger.error(f'Inference error: {completion_info.exception}')
            else:
                bindings = bindings_list[0]
                if len(bindings._output_names) == 1:
                    inference_results = bindings.output().get_buffer()
                else:
                    inference_results = {
                        name: np.expand_dims(bindings.output(name).get_buffer(), axis=0)
                        for name in bindings._output_names
                    }
        
        hailo_inference.run_sync([preprocessed_frame], callback)
        
        if inference_results is not None:
            try:
                output_queue.put((frame, inference_results), timeout=1)
            except queue.Full:
                pass
        
        input_queue.task_done()
    logger.info("Inference thread exiting.")

def main():
    parser = argparse.ArgumentParser(description="Hailo Object Detection")
    parser.add_argument("--net-path", type=str, default="object_detection/yolov8n.hef", help="Path to the HEF file")
    parser.add_argument("--labels-path", type=str, default="coco.txt", help="Path to the labels file")
    parser.add_argument("--config-path", type=str, default="config.json", help="Path to the config file")
    args = parser.parse_args()
    
    labels = get_labels(args.labels_path)
    config_data = load_json_file(args.config_path)
    
    # Initialize Tracker
    tracker_config = config_data.get("visualization_params", {}).get("tracker", {})
    tracker = BYTETracker(SimpleNamespace(**tracker_config))
    
    # Initialize Hailo Inference
    hailo_inference = HailoInfer(args.net_path, batch_size=1)
    model_h, model_w, _ = hailo_inference.get_input_shape()
    
    # Initialize Picamera2
    picam = picamera2.Picamera2()
    stream_config = picam.create_preview_configuration(main={"size": (1280, 720)})
    picam.configure(stream_config)
    picam.start()
    
    # Queues
    input_queue = queue.Queue(maxsize=3)
    output_queue = queue.Queue(maxsize=3)
    stop_event = threading.Event()
    
    # Start Threads
    capture_thread = threading.Thread(target=capture_thread_func, args=(picam, model_w, model_h, input_queue, stop_event))
    inference_thread = threading.Thread(target=inference_thread_func, args=(hailo_inference, input_queue, output_queue, stop_event))
    
    capture_thread.start()
    inference_thread.start()
    
    fps_tracker = FrameRateTracker()
    fps_tracker.start()
    
    logger.info("Starting UI loop. Press 'q' to quit.")
    
    try:
        cv2.namedWindow("Hailo Object Detection", cv2.WINDOW_NORMAL)
        
        while not stop_event.is_set():
            try:
                item = output_queue.get(timeout=1)
            except queue.Empty:
                continue
                
            frame, inference_results = item
            
            # Post-process & Track (in main thread to avoid complex sync with tracker)
            detections = extract_detections(frame, inference_results, config_data)
            annotated_frame = draw_detections(detections, frame, labels, tracker=tracker)
            
            # Display
            display_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
            fps_tracker.increment()
            
            fps_text = f"FPS: {fps_tracker.fps:.1f}"
            cv2.putText(display_frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow("Hailo Object Detection", display_frame)
            
            output_queue.task_done()
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_event.set()
                break
                
    except Exception as e:
        logger.exception(f"An error occurred: {e}")
    finally:
        logger.info("Cleaning up...")
        stop_event.set()
        capture_thread.join(timeout=2)
        inference_thread.join(timeout=2)
        picam.stop()
        cv2.destroyAllWindows()
        hailo_inference.close()
        logger.info(fps_tracker.frame_rate_summary())

if __name__ == "__main__":
    main()
