import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import os
import hashlib
import logging
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av

# --- Configuration ---
MODEL_PATHS = {
    "M1 (RoadModel_yolov8m)": "./RoadDetectionModel/RoadModel_yolov8m.pt_rounds120_b9/weights/best.pt",
    "M2 (YOLOv8_Small_2nd_Model)": "./YOLOv8_Small_2nd_Model.pt",
}
MODEL_PREFIX = {
    "M1 (RoadModel_yolov8m)": "M1",
    "M2 (YOLOv8_Small_2nd_Model)": "M2",
}
DEFAULT_CONF = 0.10
LIVE_FEED_TARGET_WIDTH = 640

# Setup logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Color Palette for Road Defect Classes (BGR format)
CLASS_COLORS = {
    "Pothole": (0, 215, 255),            # Gold / Amber
    "Potholes": (0, 215, 255),           # Gold / Amber
    "Crack": (0, 70, 255),               # Bright Orange
    "Crack-Severe": (0, 0, 255),         # Vivid Red
    "Longitudinal Crack": (0, 140, 255), # Orange
    "Transverse Crack": (0, 180, 255),   # Light Orange
    "Alligator Crack": (30, 30, 220),    # Crimson
    "Speed-Bump": (255, 220, 0),         # Cyan
    "Heavy-Vehicle": (200, 50, 200),     # Magenta
    "Light-Vehicle": (220, 100, 220),    # Purple
    "Pedestrian": (0, 230, 120),         # Bright Green
}
DEFAULT_MARK_COLOR = (0, 215, 255)


if "processed_file_id" not in st.session_state:
    st.session_state.processed_file_id = None
if "processing_complete" not in st.session_state:
    st.session_state.processing_complete = False
if "output_file_path" not in st.session_state:
    st.session_state.output_file_path = None
if "output_file_name" not in st.session_state:
    st.session_state.output_file_name = None


@st.cache_resource
def load_yolo_model(path: str):
    """Loads a YOLO model and caches it across Streamlit sessions."""
    try:
        model = YOLO(path)
        logger.info(f"Successfully loaded model from {path}")
        return model, model.names
    except Exception as e:
        st.error(f"Error loading model at {path}: {e}")
        logger.error(f"Failed to load model at {path}", exc_info=e)
        return None, {}


def detect_anomalies(
    frame: np.ndarray,
    loaded_models: dict,
    conf_thresh: float = 0.10,
    iou_thresh: float = 0.45,
    auto_enhance: bool = True,
    auto_sensitivity: bool = True,
):
    """
    Runs multi-model inference with low-resolution enhancement, Non-Maximum
    Suppression (NMS) fusion, and auto-sensitivity for subtle road anomalies.
    """
    h, w = frame.shape[:2]
    scale_factor = 1.0

    # Auto-enhance low-resolution web thumbnails (e.g. 200-300px) so YOLO sees clear road features
    if auto_enhance and max(h, w) < 550:
        target_dim = 640.0
        scale_factor = target_dim / max(h, w)
        feed_frame = cv2.resize(
            frame,
            (int(w * scale_factor), int(h * scale_factor)),
            interpolation=cv2.INTER_CUBIC,
        )
    else:
        feed_frame = frame

    def run_pass(conf_cutoff: float):
        raw_boxes = []
        raw_scores = []
        raw_meta = []

        for model_name, (model, names_map) in loaded_models.items():
            if model is None:
                continue
            try:
                results = model.predict(feed_frame, conf=conf_cutoff, verbose=False)[0]
                for b in results.boxes:
                    box_xyxy = (b.xyxy[0].cpu().numpy() / scale_factor).tolist()
                    conf = float(b.conf[0])
                    cls_id = int(b.cls[0])
                    cls_name = names_map.get(cls_id, str(cls_id))

                    bw = max(1, int(box_xyxy[2] - box_xyxy[0]))
                    bh = max(1, int(box_xyxy[3] - box_xyxy[1]))
                    bx = int(box_xyxy[0])
                    by = int(box_xyxy[1])

                    raw_boxes.append([bx, by, bw, bh])
                    raw_scores.append(conf)
                    raw_meta.append({
                        "model": MODEL_PREFIX.get(model_name, model_name),
                        "model_full": model_name,
                        "class": cls_name,
                        "conf": conf,
                        "box": [
                            max(0, bx),
                            max(0, by),
                            min(w - 1, bx + bw),
                            min(h - 1, by + bh),
                        ],
                    })
            except Exception as e:
                logger.error(f"Error during predict on {model_name}: {e}")

        if not raw_boxes:
            return []

        # NMS Fusion: merge duplicate overlapping detections across both models
        keep_indices = cv2.dnn.NMSBoxes(
            raw_boxes, raw_scores, score_threshold=conf_cutoff, nms_threshold=iou_thresh
        )

        fused = []
        if len(keep_indices) > 0:
            for idx in keep_indices:
                fused.append(raw_meta[int(idx)])
        return fused

    # Pass 1: standard user threshold
    detections = run_pass(conf_thresh)
    is_faint = False

    # Pass 2: Auto-sensitivity fallback if nothing detected at current threshold
    if not detections and auto_sensitivity and conf_thresh > 0.08:
        detections = run_pass(0.08)
        if detections:
            is_faint = True

    return detections, is_faint


def render_anomalies(
    frame: np.ndarray,
    detections: list,
    mark_style: str = "Target Circle & Box",
    is_faint: bool = False,
):
    """
    Renders high-visibility marks (circles, boxes, soft highlight tint, badges)
    scaled adaptively to image dimensions.
    """
    out = frame.copy()
    if not detections:
        return out

    h, w = out.shape[:2]
    base_dim = max(h, w)
    dyn_thickness = max(2, int(base_dim / 300))
    dyn_font_scale = max(0.45, base_dim / 1100)

    # 1. Soft translucent highlight overlay over defect regions
    overlay = out.copy()
    for d in detections:
        x1, y1, x2, y2 = [int(v) for v in d["box"]]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        color = CLASS_COLORS.get(d["class"], DEFAULT_MARK_COLOR)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, 0.20, out, 0.80, 0, out)

    # 2. Draw circles, boxes, and badges
    for d in detections:
        x1, y1, x2, y2 = [int(v) for v in d["box"]]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        color = CLASS_COLORS.get(d["class"], DEFAULT_MARK_COLOR)

        bw = x2 - x1
        bh = y2 - y1
        cx, cy = x1 + bw // 2, y1 + bh // 2
        radius = int(np.sqrt((bw / 2) ** 2 + (bh / 2) ** 2))

        # Circle marker: yellow outer ring + center target dot
        if "Circle" in mark_style:
            cv2.circle(out, (cx, cy), radius, (0, 255, 255), dyn_thickness)
            cv2.circle(out, (cx, cy), max(2, dyn_thickness), color, -1)

        # Bounding box marker
        if "Box" in mark_style:
            cv2.rectangle(out, (x1, y1), (x2, y2), color, dyn_thickness)

        # Label badge
        label_prefix = f"[{d['model']}] " if "model" in d else ""
        label_text = f"{label_prefix}{d['class']} {d['conf']*100:.0f}%"
        if is_faint and d["conf"] < 0.15:
            label_text += " (Faint)"

        (tw, th), bl = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, dyn_font_scale, 1
        )
        badge_y1 = max(0, y1 - th - 8)
        badge_y2 = y1
        badge_x2 = min(w - 1, x1 + tw + 8)

        cv2.rectangle(out, (x1, badge_y1), (badge_x2, badge_y2), color, -1)
        cv2.putText(
            out,
            label_text,
            (x1 + 4, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            dyn_font_scale,
            (0, 0, 0),
            max(1, int(dyn_thickness / 2)),
            cv2.LINE_AA,
        )

    return out


def cleanup_previous_output():
    """Deletes temporary output files from previous video runs."""
    if st.session_state.output_file_path and os.path.exists(
        st.session_state.output_file_path
    ):
        try:
            os.remove(st.session_state.output_file_path)
            logger.info(
                f"Cleaned up previous output file: {st.session_state.output_file_path}"
            )
        except OSError as rm_err:
            logger.error(
                f"Error removing previous output file {st.session_state.output_file_path}: {rm_err}"
            )
    st.session_state.output_file_path = None
    st.session_state.output_file_name = None
    st.session_state.processing_complete = False
    st.session_state.processed_file_id = None


def handle_image_input(loaded_models, conf_thresh, mark_style, auto_sensitivity):
    """Handles image upload, cached detection, side-by-side rendering, and metrics."""
    if st.session_state.processed_file_id is not None:
        cleanup_previous_output()

    st.subheader("📷 Road Anomaly Detection (Image)")
    st.write(
        "Upload any road surface image to spot **potholes, cracks, severe fissures, and speed bumps**."
    )

    uploaded_file = st.file_uploader(
        "Choose a road photo (JPG, PNG, WEBP)",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        key="img_upload_main",
    )

    if uploaded_file is not None:
        try:
            file_bytes = uploaded_file.getvalue()
        except Exception as read_err:
            st.error(f"Failed to read image file: {read_err}")
            return

        if not file_bytes:
            st.error("Uploaded file is empty.")
            return

        # Stable session cache key: ensures results remain fixed without flickering on reruns
        file_hash = hashlib.md5(file_bytes).hexdigest()
        cache_key = f"{file_hash}_{conf_thresh}_{mark_style}_{auto_sensitivity}_{sorted(loaded_models.keys())}"

        if (
            "image_cache" in st.session_state
            and st.session_state.image_cache.get("key") == cache_key
        ):
            # Instant load from cache
            cached_data = st.session_state.image_cache
            original_rgb = cached_data["original_rgb"]
            processed_rgb = cached_data["processed_rgb"]
            processed_bgr = cached_data["processed_bgr"]
            detections = cached_data["detections"]
            is_faint = cached_data["is_faint"]
        else:
            arr = np.frombuffer(file_bytes, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

            if img is None:
                st.error("Could not decode image. Please upload a valid image file.")
                return

            with st.spinner("🔍 Running YOLOv8 dual-model analysis..."):
                detections, is_faint = detect_anomalies(
                    frame=img,
                    loaded_models=loaded_models,
                    conf_thresh=conf_thresh,
                    iou_thresh=0.45,
                    auto_enhance=True,
                    auto_sensitivity=auto_sensitivity,
                )
                processed_bgr = render_anomalies(
                    frame=img,
                    detections=detections,
                    mark_style=mark_style,
                    is_faint=is_faint,
                )

            original_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            processed_rgb = cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2RGB)

            st.session_state.image_cache = {
                "key": cache_key,
                "original_rgb": original_rgb,
                "processed_rgb": processed_rgb,
                "processed_bgr": processed_bgr,
                "detections": detections,
                "is_faint": is_faint,
            }
            cached_data = st.session_state.image_cache

        # Side-by-side display
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 🖼️ Original Image")
            st.image(original_rgb, use_container_width=True)

        with col2:
            st.markdown(f"#### 🎯 Detected Anomalies ({mark_style})")
            st.image(processed_rgb, use_container_width=True)

        st.markdown("---")

        if detections:
            if is_faint:
                st.info(
                    f"💡 **Auto-Sensitivity Activated**: Detected **{len(detections)}** subtle/faint road anomalies "
                    f"(scores between 8% - {conf_thresh*100:.0f}%). Marked with circles."
                )
            else:
                st.success(f"✅ Found **{len(detections)}** road anomaly/anomalies!")

            # Class count cards
            class_counts = {}
            for d in detections:
                class_counts[d["class"]] = class_counts.get(d["class"], 0) + 1

            metric_cols = st.columns(min(max(len(class_counts), 1), 4))
            for i, (cls_name, count) in enumerate(class_counts.items()):
                with metric_cols[i % len(metric_cols)]:
                    st.metric(label=cls_name, value=count)

            # Details table
            table_data = [
                {
                    "Defect Type": d["class"],
                    "Confidence": f"{d['conf']*100:.1f}%",
                    "Model": d.get("model_full", d.get("model", "YOLOv8")),
                    "Location [X1, Y1, X2, Y2]": str([int(x) for x in d["box"]]),
                }
                for d in detections
            ]
            with st.expander("📋 View Detection Details Table", expanded=True):
                st.dataframe(table_data, use_container_width=True)

            # Download annotated image button
            success, encoded_img = cv2.imencode(".jpg", cached_data["processed_bgr"])
            if success:
                st.download_button(
                    label="⬇️ Download Marked Image",
                    data=encoded_img.tobytes(),
                    file_name=f"detected_{uploaded_file.name}",
                    mime="image/jpeg",
                    key="download_annotated_img",
                )
        else:
            st.warning(
                "⚠️ **No road anomalies detected** with current settings.\n\n"
                "💡 **Tips:**\n"
                "- Try moving the **Confidence Threshold slider** in the sidebar down to **0.05**.\n"
                "- Ensure **Both Models (M1 + M2)** are selected in the sidebar.\n"
                "- Keep **Auto-Sensitivity** enabled."
            )
    else:
        st.info("👆 Please upload a road image using the box above to begin detection.")


def handle_video_input(loaded_models, conf_thresh, mark_style):
    """Processes uploaded video files frame-by-frame with progress reporting."""
    uploaded_file = st.sidebar.file_uploader(
        "Upload Video", type=["mp4", "avi", "mov", "mkv"], key="vid_upload"
    )

    if uploaded_file:
        current_file_id = uploaded_file.file_id

        if current_file_id != st.session_state.processed_file_id:
            logger.info(f"New video uploaded (ID: {current_file_id}). Resetting state.")
            cleanup_previous_output()
            st.session_state.processed_file_id = current_file_id

        if st.session_state.processing_complete and st.session_state.output_file_path:
            st.success("✅ Video processing complete!")
            if os.path.exists(st.session_state.output_file_path):
                try:
                    with open(st.session_state.output_file_path, "rb") as f:
                        video_bytes = f.read()
                    st.download_button(
                        label="⬇️ Download Processed Video",
                        data=video_bytes,
                        file_name=st.session_state.output_file_name
                        or f"processed_{uploaded_file.name}",
                        mime="video/mp4",
                        key="download_btn_rerun",
                    )
                except Exception as e:
                    st.error(f"Error reading processed video: {e}")
            else:
                st.error("Processed file missing. Please upload again.")
                cleanup_previous_output()
            return

        input_tmp_path = None
        output_video_path = None
        cap = None
        writer = None
        processing_succeeded = False

        try:
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=".mp4"
            ) as input_tmp_file:
                input_tmp_path = input_tmp_file.name
                input_tmp_file.write(uploaded_file.getvalue())

            cap = cv2.VideoCapture(input_tmp_path)
            if not cap.isOpened():
                st.error("Error opening uploaded video file.")
                return

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            with tempfile.NamedTemporaryFile(
                delete=False, suffix=".mp4"
            ) as output_tmp_file:
                output_video_path = output_tmp_file.name

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
            if not writer.isOpened():
                st.error("Failed to initialize video writer.")
                return

            prog_bar = st.progress(0, text="Analyzing video frames...")
            frame_idx = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                dets, is_faint = detect_anomalies(
                    frame=frame,
                    loaded_models=loaded_models,
                    conf_thresh=conf_thresh,
                    auto_enhance=False,
                    auto_sensitivity=False,
                )
                out_frame = render_anomalies(
                    frame=frame,
                    detections=dets,
                    mark_style=mark_style,
                    is_faint=is_faint,
                )
                writer.write(out_frame)
                frame_idx += 1
                progress = frame_idx / total_frames if total_frames > 0 else 0
                prog_bar.progress(
                    min(progress, 1.0),
                    text=f"Processing frame {frame_idx}/{total_frames if total_frames > 0 else '?'}",
                )

            processing_succeeded = True
            prog_bar.progress(1.0, text="Video analysis finished.")

        except Exception as e:
            st.error(f"Error during video processing: {e}")
            logger.error("Error in video loop", exc_info=e)

        finally:
            if cap is not None:
                cap.release()
            if writer is not None:
                writer.release()
            if input_tmp_path and os.path.exists(input_tmp_path):
                try:
                    os.remove(input_tmp_path)
                except OSError:
                    pass

        if processing_succeeded and output_video_path and os.path.exists(output_video_path):
            st.session_state.output_file_path = output_video_path
            st.session_state.output_file_name = f"detected_{uploaded_file.name}"
            st.session_state.processing_complete = True
            st.session_state.processed_file_id = current_file_id

            st.success("✅ Video processing complete!")
            with open(st.session_state.output_file_path, "rb") as f:
                st.download_button(
                    label="⬇️ Download Processed Video",
                    data=f.read(),
                    file_name=st.session_state.output_file_name,
                    mime="video/mp4",
                    key="download_btn_first",
                )
    else:
        if st.session_state.processed_file_id is not None:
            cleanup_previous_output()
        st.info("Upload a video using the sidebar to start.")


class YOLOVideoProcessor(VideoProcessorBase):
    def __init__(self, loaded_models, conf_thresh, mark_style, target_width):
        self.loaded_models = loaded_models
        self.conf_thresh = conf_thresh
        self.mark_style = mark_style
        self.target_width = target_width

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        h, w = img.shape[:2]
        if self.target_width and w > self.target_width:
            img = cv2.resize(img, (self.target_width, int(h * self.target_width / w)))

        dets, is_faint = detect_anomalies(
            frame=img,
            loaded_models=self.loaded_models,
            conf_thresh=self.conf_thresh,
            auto_enhance=False,
            auto_sensitivity=False,
        )
        annotated = render_anomalies(
            frame=img,
            detections=dets,
            mark_style=self.mark_style,
            is_faint=is_faint,
        )
        return av.VideoFrame.from_ndarray(annotated, format="bgr24")


def handle_live_camera(loaded_models, conf_thresh, mark_style):
    if st.session_state.processed_file_id is not None:
        cleanup_previous_output()

    st.sidebar.info("Grant camera permissions in your browser and click START.")
    media_constraints = {
        "video": {"width": {"ideal": 640}, "height": {"ideal": 480}, "frameRate": {"ideal": 15}},
        "audio": False,
    }

    def factory():
        return YOLOVideoProcessor(
            loaded_models=loaded_models,
            conf_thresh=conf_thresh,
            mark_style=mark_style,
            target_width=LIVE_FEED_TARGET_WIDTH,
        )

    webrtc_ctx = webrtc_streamer(
        key="live-camera-streamer",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=factory,
        media_stream_constraints=media_constraints,
        async_processing=True,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    )
    if not webrtc_ctx.state.playing:
        st.info("Camera feed stopped or waiting to start.")


def main():
    st.set_page_config(layout="wide", page_title="Road Anomaly Detection AI")
    st.title("🚗 Road Anomaly Detection (AI Vision System)")
    st.caption("AI-powered surface inspection for Potholes, Cracks, Severe Damage & Hazards.")

    # --- Sidebar Configuration ---
    st.sidebar.header("⚙️ Model Configuration")

    st.sidebar.subheader("🧠 Model Selection")
    model_choice = st.sidebar.radio(
        "Detection Engine",
        ["Ensemble (Both Models - Recommended)", "Model 1 Only (best.pt)", "Model 2 Only (YOLOv8s)"],
        index=0,
    )

    models_to_load = {}
    if "Both" in model_choice or "Model 1" in model_choice:
        models_to_load["M1 (RoadModel_yolov8m)"] = MODEL_PATHS["M1 (RoadModel_yolov8m)"]
    if "Both" in model_choice or "Model 2" in model_choice:
        models_to_load["M2 (YOLOv8_Small_2nd_Model)"] = MODEL_PATHS["M2 (YOLOv8_Small_2nd_Model)"]

    loaded_models = {}
    for name, path in models_to_load.items():
        m, names = load_yolo_model(path)
        if m:
            loaded_models[name] = (m, names)

    if not loaded_models:
        st.sidebar.error("Failed to load model weights. Ensure .pt files exist.")
        st.stop()

    st.sidebar.subheader("🎚️ Detection Sensitivity")
    conf_thresh = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.05,
        max_value=0.90,
        value=0.10,
        step=0.05,
        help="Lower confidence (0.05 - 0.15) detects faint cracks and shallow potholes.",
    )

    auto_sensitivity = st.sidebar.checkbox(
        "Auto-Sensitivity Fallback",
        value=True,
        help="If nothing is detected at current threshold, automatically scans for subtle defects down to 8% confidence.",
    )

    st.sidebar.subheader("🎨 Anomaly Marking Style")
    mark_style = st.sidebar.radio(
        "Visual Marker",
        ["Target Circle & Box", "Circle Only", "Box Only"],
        index=0,
        help="Choose how detected potholes and cracks are highlighted on the image.",
    )

    st.sidebar.subheader("🎬 Input Source")
    input_mode = st.sidebar.radio(
        "Select Input Source", ["Image", "Video", "Live Dashcam"], key="input_mode_radio"
    )

    if input_mode == "Image":
        handle_image_input(loaded_models, conf_thresh, mark_style, auto_sensitivity)
    elif input_mode == "Video":
        handle_video_input(loaded_models, conf_thresh, mark_style)
    elif input_mode == "Live Dashcam":
        handle_live_camera(loaded_models, conf_thresh, mark_style)

    st.markdown("---")
    st.caption("Road Anomaly Detection System • Powered by YOLOv8 Deep Learning")


if __name__ == "__main__":
    main()
