import sounddevice as sd
import numpy as np
import os, time, cv2, whisper, math
from gtts import gTTS
import pygame
from pydub import AudioSegment
from ultralytics import YOLO
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from pprint import pprint


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

# Parameters
SAMPLE_RATE = 16000
SILENCE_THRESHOLD = 0.003
CHUNK_DURATION = 1
SILENCE_TOLERANCE = 2
AUDIO_BUFFER = []

def speak(message):
    """Convert text to speech using Google Text-to-Speech."""
    temp_file = "temp_speech.mp3"
    try:
        # Create gTTS object
        tts = gTTS(text=message, lang='en', slow=False)
        
        # Save to temporary file
        tts.save(temp_file)
        
        # Initialize pygame mixer
        pygame.mixer.init()
        
        # Load and play the audio
        pygame.mixer.music.load(temp_file)
        pygame.mixer.music.play()
        
        # Wait for the audio to finish playing
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
            
        # Clean up
        pygame.mixer.quit()  # Fully quit pygame mixer
        
        # Ensure file is closed and deleted
        if os.path.exists(temp_file):
            os.remove(temp_file)
            
    except Exception as e:
        print(f"Error in text-to-speech conversion: {e}")
    finally:
        # Make absolutely sure the file is deleted even if an error occurs
        if os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except:
                pass


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
    rtsp_url = (
        f"rtsp://{username}:{password}@{camera_ip}:554"
        if username and password
        else f"rtsp://{camera_ip}:554"
    )
    print(f"Using RTSP URL: {rtsp_url}")

    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        print(
            "Error: Unable to connect to the camera. Check the RTSP URL or network connection."
        )
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
                    (x1, y1, x2, y2), frame_width, frame_height
                )

                matched_objects.append(
                    {
                        "class_name": class_name,
                        "confidence": float(confidence),
                        "pixel_coords": (x1, y1, x2, y2),
                        "real_world_coords": {
                            "x": round(real_x, 2),
                            "y": round(real_y, 2),
                            "depth": round(depth, 2),
                        },
                    }
                )

                # Draw bounding box and coordinates
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{class_name} ({real_x:.1f}m, {real_y:.1f}m, {depth:.1f}m)"
                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    2,
                )

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
        cv2.imshow("Real-Time Object Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        # Check for timeout
        elapsed_time = time.time() - start_time
        if elapsed_time > search_timeout:
            print(
                f"Timeout reached ({search_timeout} seconds). Returning to speech-to-text."
            )
            break

    cap.release()
    cv2.destroyAllWindows()
    return False


def extract_objects(input_text, detected_objects):
    """
    Process user queries about objects and compare with detected objects.
    Args:
        input_text: User's speech converted to text
        detected_objects: List of objects currently detected by the camera
    Returns:
        str: Natural language response about objects
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a helpful assistant that answers questions about objects in view of a camera.
            Compare the user's question with the list of detected objects and respond naturally.
            
            Rules:
            - For "what can you see?" questions, list all detected objects in a natural way
            - For specific object queries ("can you see X?"), confirm if the object is detected
            - If asked about an object that isn't detected, clearly state it's not visible
            - Keep responses conversational but concise""",
            ),
            (
                "human",
                """User question: {input_text}
            Currently detected objects: {detected_objects}
            
            Provide a natural response:""",
            ),
        ]
    )

    try:
        chain = prompt | llm
        response = chain.invoke(
            {"input_text": input_text, "detected_objects": detected_objects}
        )
        return response.content
    except Exception as e:
        print(f"Error processing query: {e}")
        return "I'm having trouble processing that request."


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


def handle_spatial_query_locally(input_text, detected_objects_with_coords):
    """
    Handles basic spatial queries locally without LLM.
    Args:
        input_text: The user's query in text form.
        detected_objects_with_coords: List of detected objects with their coordinates.
    Returns:
        str: Response to the query.
    """
    keywords = {
        "left": lambda obj: obj["coordinates"]["x"],
        "right": lambda obj: obj["coordinates"]["x"],
        "closer": lambda obj: obj["coordinates"]["depth"],
        "farther": lambda obj: obj["coordinates"]["depth"],
        "higher": lambda obj: obj["coordinates"]["y"],
        "lower": lambda obj: obj["coordinates"]["y"],
    }

    for key, sort_func in keywords.items():
        if key in input_text.lower():
            sorted_objs = sorted(
                detected_objects_with_coords,
                key=sort_func,
                reverse=(key in ["right", "farther", "lower"]),
            )
            if len(sorted_objs) > 1:
                return f"The {sorted_objs[0]['class_name']} is {key} than the {sorted_objs[1]['class_name']}."
            elif sorted_objs:
                return (
                    f"The only detected object is the {sorted_objs[0]['class_name']}."
                )
            else:
                return "No objects detected to compare."

    return None  # Indicates query is too complex for local handling


def process_spatial_query(input_text, detected_objects_with_coords):
    """
    Process spatial queries about object locations.
    Args:
        input_text: User's speech converted to text
        detected_objects_with_coords: List of dictionaries containing object info and coordinates
    Returns:
        str: Natural language response about object locations
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a helpful assistant that answers questions about the spatial relationships between objects in view of a camera.
            The coordinates provided are in meters where:
            - x: horizontal position (negative is left, positive is right)
            - y: vertical position (negative is lower, positive is higher)
            - depth: distance from camera (larger numbers are further away)

            Rules:
            - For questions about left/right, compare x coordinates
            - For questions about higher/lower, compare y coordinates
            - For questions about closer/farther, compare depth values
            - Use natural language to describe relative positions
            - Be specific about which object is where
            - Keep responses conversational but precise""",
            ),
            (
                "human",
                """User question: {input_text}
            Detected objects with coordinates: {detected_objects_with_coords}
            
            Provide a natural response describing the spatial relationships:""",
            ),
        ]
    )

    try:
        chain = prompt | llm
        response = chain.invoke(
            {
                "input_text": input_text,
                "detected_objects_with_coords": detected_objects_with_coords,
            }
        )
        return response.content
    except Exception as e:
        print(f"Error processing spatial query: {e}")
        return "I'm having trouble processing that spatial request."


def main():
    """Main execution flow for the object detection and query system."""
    while True:
        # Get speech input
        speech_text = record_audio()
        print(f"You said: {speech_text}")

        # Get current camera view and detect objects
        detected_objects = []
        detected_objects_with_coords = []
        cap = get_camera_stream()

        if cap is not None:
            ret, frame = cap.read()
            if ret:
                frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                results = object_model(frame)

                # Process detected objects
                for box in results[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    class_name = object_model.names[int(box.cls[0])]

                    # Get real-world coordinates
                    real_x, real_y, depth = pixel_to_real_world(
                        (x1, y1, x2, y2), frame_width, frame_height
                    )

                    detected_objects.append(class_name)
                    detected_objects_with_coords.append(
                        {
                            "class_name": class_name,
                            "coordinates": {
                                "x": round(real_x, 2),
                                "y": round(real_y, 2),
                                "depth": round(depth, 2),
                            },
                        }
                    )

                detected_objects = list(set(detected_objects))
                cap.release()

        # Check if the question is about spatial relationships
        spatial_keywords = [
            "left",
            "right",
            "above",
            "below",
            "higher",
            "lower",
            "closer",
            "farther",
            "near",
            "far",
            "where",
            "location",
        ]

        if any(keyword in speech_text.lower() for keyword in spatial_keywords):
            response = handle_spatial_query_locally(
                speech_text, detected_objects_with_coords
            )
            if response is None:  # Fallback to LLM for complex queries
                response = process_spatial_query(
                    speech_text, detected_objects_with_coords
                )
        else:
            response = extract_objects(speech_text, detected_objects)

        print(f"Assistant: {response}")
        speak(response)


if __name__ == "__main__":
    main()
