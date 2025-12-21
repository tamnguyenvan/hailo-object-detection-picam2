import cv2
import numpy as np
import sys
import os
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
from object_detection_post_process import extract_detections, draw_detections, inference_result_handler
from tracker.byte_tracker import BYTETracker
from toolbox import (
    get_labels,
    load_json_file,
    default_preprocess,
    FrameRateTracker,
)

def main():
    parser = argparse.ArgumentParser(description="Hailo Object Detection")
    parser.add_argument("--net-path", type=str, default="object_detection/yolov8n.hef", help="Path to the HEF file")
    parser.add_argument("--labels-path", type=str, default="coco.txt", help="Path to the labels file")
    parser.add_argument("--config-path", type=str, default="config.json", help="Path to the config file")
    args = parser.parse_args()
    
    # Configuration
    net_path = args.net_path
    labels_path = args.labels_path
    config_path = args.config_path
    
    labels = get_labels(labels_path)
    config_data = load_json_file(config_path)
    
    # Initialize Tracker
    tracker_config = config_data.get("visualization_params", {}).get("tracker", {})
    tracker = BYTETracker(SimpleNamespace(**tracker_config))
    
    # Initialize Hailo Inference
    hailo_inference = HailoInfer(net_path, batch_size=1)
    model_h, model_w, _ = hailo_inference.get_input_shape()
    
    # Initialize Picamera2
    picam = picamera2.Picamera2()
    # Configure for HD resolution (or as needed)
    stream_config = picam.create_preview_configuration(main={"size": (1280, 720)})
    picam.configure(stream_config)
    picam.start()
    
    fps_tracker = FrameRateTracker()
    fps_tracker.start()
    
    logger.info("Starting inference loop. Press 'q' to quit.")
    
    try:
        cv2.namedWindow("Hailo Object Detection", cv2.WINDOW_NORMAL)
        
        while True:
            # 1. Capture frame
            frame = picam.capture_array()
            # frame is RGB (Picamera2 default), convert to BGR for OpenCV display
            # but keep RGB for preprocessing if model expects it
            
            # 2. Preprocess
            # default_preprocess expects RGB, returns RGB padded
            preprocessed_frame = default_preprocess(frame, model_w, model_h)
            
            # 3. Inference (Sync)
            inference_results = None
            def callback(completion_info, bindings_list, input_batch, output_queue):
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
            
            # 4. Post-process & Track
            if inference_results is not None:
                # extract_detections expects BGR or RGB? 
                # Let's check object_detection_post_process.py: it uses cv2.rectangle which works on any.
                # It expects original image for size.
                detections = extract_detections(frame, inference_results, config_data)
                
                # draw_detections handles tracking if tracker is provided
                # It returns the annotated frame
                annotated_frame = draw_detections(detections, frame, labels, tracker=tracker)
                
                # 5. Display
                display_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
                fps_tracker.increment()
                
                # Add FPS overlay
                fps_text = f"FPS: {fps_tracker.fps:.1f}"
                cv2.putText(display_frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                cv2.imshow("Hailo Object Detection", display_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except Exception as e:
        logger.exception(f"An error occurred: {e}")
    finally:
        logger.info("Cleaning up...")
        picam.stop()
        cv2.destroyAllWindows()
        hailo_inference.close()
        logger.info(fps_tracker.frame_rate_summary())

if __name__ == "__main__":
    main()
