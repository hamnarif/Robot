import sounddevice as sd
import numpy as np
import os, time, ast, re, asyncio, cv2, whisper, pyttsx3, math
from pydub import AudioSegment
from ultralytics import YOLO
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from pprint import pprint
from dotenv import load_dotenv

# Load environment variables for camera at the start
load_dotenv()
# username = os.getenv("CAMERA_USERNAME")
# password = os.getenv("CAMERA_PASSWORD")
# camera_ip = os.getenv("CAMERA_IP", "192.168.1.64")  # Default IP if not in .env

username = "admin"
password = 'SNO"2025'
camera_ip = "169.254.91.58" 

print(f"CAMERA_USERNAME: {username}")
print(f"CAMERA_PASSWORD: {password}")
print(f"CAMERA_IP: {camera_ip}")

# Initialize models
speech_model = whisper.load_model("base")
object_model = YOLO("yolo11x.pt")
llm = ChatOllama(model="llama3.1")

# Initialize text-to-speech engine
tts_engine = pyttsx3.init()

# Parameters
SAMPLE_RATE = 16000
SILENCE_THRESHOLD = 0.001
CHUNK_DURATION = 1
SILENCE_TOLERANCE = 4
AUDIO_BUFFER = []

def speak(message):
    """Convert text to speech."""
    tts_engine.say(message)
    tts_engine.runAndWait()

# Speech Detection
def detect_speech(audio_chunk):
    """Detect if audio contains speech based on RMS."""
    rms = np.sqrt(np.mean(audio_chunk**2))
    return rms > SILENCE_THRESHOLD


def save_audio_and_transcribe(audio_data):
    """Save buffered audio to a WAV file and transcribe it."""
    print("Processing detected speech...")
    file_name = "detected_speech.wav"
    audio_segment = AudioSegment(
        data=(np.concatenate(audio_data) * 32767).astype(np.int16).tobytes(),
        sample_width=2,
        frame_rate=SAMPLE_RATE,
        channels=1,
    )
    audio_segment.export(file_name, format="wav")

    result = speech_model.transcribe(file_name)
    os.remove(file_name)
    return result["text"]


def record_audio():
    """Continuously record and process speech with silence tolerance."""
    global AUDIO_BUFFER
    print("Listening for speech...")
    silence_start_time = None

    while True:
        audio_chunk = sd.rec(
            int(CHUNK_DURATION * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
        )
        sd.wait()

        if detect_speech(audio_chunk):
            AUDIO_BUFFER.append(audio_chunk)
            silence_start_time = None
        else:
            if AUDIO_BUFFER:
                if silence_start_time is None:
                    silence_start_time = time.time()
                if time.time() - silence_start_time >= SILENCE_TOLERANCE:
                    text = save_audio_and_transcribe(AUDIO_BUFFER)
                    AUDIO_BUFFER = []
                    return text


def get_camera_stream():
    """Initialize and return camera stream."""
    print(f"Attempting to connect to camera at IP: {camera_ip}")
    
    # Construct RTSP URL
    rtsp_url = f"rtsp://{username}:{password}@{camera_ip}:554" if username and password else f"rtsp://{camera_ip}:554"
    print(f"Using RTSP URL: {rtsp_url}")
    
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        print("Error: Unable to connect to the camera. Check the RTSP URL or network connection.")
        return None
        
    print("Successfully connected to the camera.")
    return cap


def detect_objects_realtime(extracted_objects, search_timeout=15):
    """
    Detect objects in real-time from the camera feed and match with extracted objects.
    Stops searching after a timeout and returns to speech-to-text.
    """
    cap = get_camera_stream()
    if cap is None:
        print("Camera initialization failed.")
        return False

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("Starting real-time object detection. Press 'q' to exit.")

    announced_missing = False
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to retrieve frame. Check camera connection.")
            break

        results = object_model(frame)
        matched_objects = []

        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            confidence = box.conf[0]
            class_id = box.cls[0]
            class_name = object_model.names[int(class_id)]

            if class_name in extracted_objects:
                real_x, real_y, depth = pixel_to_real_world(
                    (x1, y1, x2, y2), 
                    frame_width, 
                    frame_height
                )

                matched_objects.append({
                    'class_name': class_name,
                    'confidence': float(confidence),
                    'pixel_coords': (x1, y1, x2, y2),
                    'real_world_coords': {
                        'x': round(real_x, 2),
                        'y': round(real_y, 2),
                        'depth': round(depth, 2)
                    }
                })

                # Draw bounding box and coordinates
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{class_name} ({real_x:.1f}m, {real_y:.1f}m, {depth:.1f}m)"
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        if matched_objects:
            print("\nMatched Objects with Real-World Coordinates:")
            pprint(matched_objects)
            cap.release()
            cv2.destroyAllWindows()
            return True

        # Check if no objects matched and it hasn't been announced yet
        if not matched_objects and extracted_objects and not announced_missing:
            for obj in extracted_objects:
                speak(f"I can't see the {obj}")
            announced_missing = True

        # Display the frame and handle keypress for exit
        cv2.imshow('Real-Time Object Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Check for timeout
        elapsed_time = time.time() - start_time
        if elapsed_time > search_timeout:
            print(f"Timeout reached ({search_timeout} seconds). Returning to speech-to-text.")
            break

    cap.release()
    cv2.destroyAllWindows()
    return False


def pixel_to_real_world(pixel_coords, frame_width, frame_height):
    """
    Convert pixel coordinates to real-world coordinates (in meters).
    This is a simplified conversion - you may need to calibrate for your specific setup.
    """
    # Constants for conversion (these should be calibrated for your camera setup)
    HORIZONTAL_FOV = 60  # degrees
    # VERTICAL_FOV = 45  # degrees
    TYPICAL_ROOM_WIDTH = 5.0  # meters
    TYPICAL_ROOM_HEIGHT = 4.0  # meters

    x1, y1, x2, y2 = pixel_coords

    # Calculate center point of the object
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    # Convert to normalized coordinates (-1 to 1)
    norm_x = (center_x - frame_width / 2) / (frame_width / 2)
    norm_y = (center_y - frame_height / 2) / (frame_height / 2)

    # Convert to real world coordinates
    real_x = norm_x * (TYPICAL_ROOM_WIDTH / 2)
    real_y = norm_y * (TYPICAL_ROOM_HEIGHT / 2)

    # Calculate approximate depth (z) based on object size
    object_width_pixels = x2 - x1
    normalized_size = object_width_pixels / frame_width
    depth = TYPICAL_ROOM_WIDTH / (
        2 * math.tan(math.radians(HORIZONTAL_FOV / 2)) * normalized_size
    )

    return real_x, real_y, depth


# Natural Language Processing
def extract_objects(input_text):
    """Extract objects mentioned in the text using the LLM."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Extract objects from the following text. Return the result as a Python list of strings. "
                "For example, given the input 'Pick up the book from the shelf', the output should be "
                "['book', 'shelf']. Avoid adding any extra text or explanation.",
            ),
            ("human", "{input_text}"),
        ]
    )

    try:
        chain = prompt | llm
        response = chain.invoke({"input_text": input_text})
        match = re.search(r"\[.*?\]", response.content, re.DOTALL)
        if match:
            return ast.literal_eval(match.group(0))
        return []
    except Exception as e:
        print(f"Error extracting objects: {e}")
        return []


# Main Workflow
async def main():
    while True:
        # Step 1: Detect speech and transcribe to text
        transcribed_text = record_audio()
        print("Transcribed Text:")
        print(transcribed_text)
        print()

        # Step 2: Extract objects from transcribed text
        extracted_objects = extract_objects(transcribed_text)
        print("Extracted Objects:")
        pprint(extracted_objects)
        print()

        # Step 3: Detect objects with a timeout
        detection_result = detect_objects_realtime(extracted_objects, search_timeout=15)

        # If no objects are detected, restart the workflow
        if not detection_result:
            print("No objects detected. Restarting speech-to-text process.")
            continue  # Restart the loop for new input
        else:
            print("Object detection successful. Ending program.")
            break  # Exit the loop if objects are detected

if __name__ == "__main__":
    asyncio.run(main())
