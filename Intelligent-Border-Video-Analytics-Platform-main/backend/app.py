import logging
from flask import Flask, Response, jsonify
from flask_cors import CORS
from modules.M6_core_api.stream_manager import VideoStream
import config

# Initialize logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("IBVAP-API")

# Initialize Flask App
app = Flask(__name__)

# Enable Cross-Origin Resource Sharing (CORS) for Vue.js Frontend connectivity
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Global Stream Manager instance
stream_engine = None

def init_engine():
    """Starts the background video processing thread."""
    global stream_engine
    try:
        logger.info(f"Initializing Video Stream from source: {config.VIDEO_SOURCE}")
        stream_engine = VideoStream(source=config.VIDEO_SOURCE)
        stream_engine.start()
        logger.info("Video Stream engine successfully started.")
    except Exception as e:
        logger.error(f"Failed to initialize stream engine: {str(e)}")
        raise e

def generate_mjpeg_stream():
    """
    Generator function that continuously yields processed video frames 
    as an MJPEG stream boundary.
    """
    while True:
        if stream_engine is None:
            break

        frame_bytes = stream_engine.get_jpeg_frame()
        if frame_bytes is not None:
            # MJPEG stream boundary payload format
            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n'
            )

# -------------------------------------------------------------------
# ROUTES / ENDPOINTS
# -------------------------------------------------------------------

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint to verify backend operational state."""
    return jsonify({
        "status": "HEALTHY",
        "service": "IBVAP-Backend",
        "version": "1.0.0"
    }), 200


@app.route('/api/v1/stream')
def video_feed():
    """
    Video streaming route. 
    Can be loaded directly inside HTML/Vue via <img src="http://localhost:5000/api/v1/stream" />
    """
    return Response(
        generate_mjpeg_stream(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/api/v1/stats', methods=['GET'])
def get_stats():
    """
    REST API endpoint for polling system telemetry.
    Returns the current human detection count, FPS, and running model metadata.
    """
    if stream_engine is None:
        return jsonify({"error": "Stream engine uninitialized"}), 503

    return jsonify({
        "status": "ONLINE" if stream_engine.is_running else "OFFLINE",
        "active_persons": stream_engine.active_persons,
        "current_fps": round(stream_engine.fps, 1),
        "model_info": {
            "name": "NVIDIA PeopleNet",
            "architecture": "ResNet-34 (DetectNet_v2)",
            "input_resolution": f"{config.INPUT_WIDTH}x{config.INPUT_HEIGHT}"
        }
    }), 200


@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500


# -------------------------------------------------------------------
# SERVER ENTRYPOINT
# -------------------------------------------------------------------

if __name__ == '__main__':
    # Initialize background execution thread before serving HTTP requests
    init_engine()

    logger.info(f"Starting IBVAP V1 Flask Server on {config.HOST}:{config.PORT}")
    
    # Run server (debug=False is required to prevent double-instantiating background threads)
    app.run(
        host=config.HOST,
        port=config.PORT,
        debug=False,
        threaded=True
    )
