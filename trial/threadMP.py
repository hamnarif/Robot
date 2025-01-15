# import sounddevice as sd
# import numpy as np
# import os, time, ast, re, cv2, whisper, threading
# from pydub import AudioSegment
# from ultralytics import YOLO
# from langchain_ollama import ChatOllama
# from langchain_core.prompts import ChatPromptTemplate
# from pprint import pprint
# from queue import Queue
# import logging
# logging.getLogger('cv2').setLevel(logging.ERROR)

# # Initialize models
# speech_model = whisper.load_model("base")
# object_model = YOLO("yolo11x.pt", verbose=False)
# llm = ChatOllama(model="llama3.1")

# # Parameters
# SAMPLE_RATE = 16000
# SILENCE_THRESHOLD = 0.001
# CHUNK_DURATION = 1
# SILENCE_TOLERANCE = 4
# AUDIO_BUFFER = []

# # Shared queues for communication between threads
# detected_objects_queue = Queue()
# transcribed_text_queue = Queue()

# # Speech Detection
# def detect_speech(audio_chunk):
#     """Detect if audio contains speech based on RMS."""
#     rms = np.sqrt(np.mean(audio_chunk ** 2))
#     return rms > SILENCE_THRESHOLD

# def save_audio_and_transcribe():
#     """Continuously record and transcribe audio."""
#     global AUDIO_BUFFER
#     print("Listening for speech...")
#     silence_start_time = None

#     while True:
#         audio_chunk = sd.rec(int(CHUNK_DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
#         sd.wait()

#         if detect_speech(audio_chunk):
#             AUDIO_BUFFER.append(audio_chunk)
#             silence_start_time = None
#         else:
#             if AUDIO_BUFFER:
#                 if silence_start_time is None:
#                     silence_start_time = time.time()
#                 if time.time() - silence_start_time >= SILENCE_TOLERANCE:
#                     print("Processing detected speech...")
#                     file_name = "detected_speech.wav"
#                     audio_segment = AudioSegment(
#                         data=(np.concatenate(AUDIO_BUFFER) * 32767).astype(np.int16).tobytes(),
#                         sample_width=2,
#                         frame_rate=SAMPLE_RATE,
#                         channels=1
#                     )
#                     audio_segment.export(file_name, format="wav")

#                     result = speech_model.transcribe(file_name)
#                     os.remove(file_name)
#                     AUDIO_BUFFER = []

#                     # Put the transcribed text into the queue
#                     transcribed_text_queue.put(result['text'])
#                     break

# # Real-Time Object Detection
# def detect_objects_realtime():
#     cap = cv2.VideoCapture(0)  # Use 0 for the default camera

#     if not cap.isOpened():
#         print("Error: Could not open the camera.")
#         return

#     print("Starting real-time object detection...")

#     frame_count = 0
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("Error: Failed to capture the frame.")
#             break

#         # Run YOLO object detection on the frame
#         results = object_model(frame)

#         # Periodically log detection summary
#         frame_count += 1
#         if frame_count % 10 == 0:
#             print(f"Frame {frame_count}: {len(results[0].boxes)} objects detected.")

#         # Draw detections on the frame
#         for box in results[0].boxes:
#             x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
#             confidence = box.conf[0]
#             class_id = box.cls[0]
#             class_name = object_model.names[int(class_id)]
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#             label = f"{class_name} {confidence:.2f}"
#             cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

#         cv2.imshow('Real-Time Object Detection', frame)

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()

# # Natural Language Processing
# def extract_objects(input_text):
#     """Extract objects mentioned in the text using the LLM."""
#     prompt = ChatPromptTemplate.from_messages([
#         (
#             "system",
#             "Extract objects from the following text. Return the result as a Python list of strings. "
#             "For example, given the input 'Pick up the book from the shelf', the output should be "
#             "['book', 'shelf']. Avoid adding any extra text or explanation.",
#         ),
#         ("human", "{input_text}"),
#     ])

#     try:
#         chain = prompt | llm
#         response = chain.invoke({"input_text": input_text})
#         match = re.search(r"\[.*?\]", response.content, re.DOTALL)
#         if match:
#             return ast.literal_eval(match.group(0))
#         return []
#     except Exception as e:
#         print(f"Error extracting objects: {e}")
#         return []

# # Main Workflow
# def main():
#     # Start threads for real-time object detection and speech-to-text transcription
#     detection_thread = threading.Thread(target=detect_objects_realtime)
#     transcription_thread = threading.Thread(target=save_audio_and_transcribe)

#     detection_thread.start()
#     transcription_thread.start()

#     # Wait for both threads to finish
#     detection_thread.join()
#     transcription_thread.join()

#     # Get the detected objects and transcribed text from the queues
#     detected_objects = detected_objects_queue.get()
#     transcribed_text = transcribed_text_queue.get()

#     print("\nTranscribed Text:")
#     print(transcribed_text)

#     # print("\nDetected Objects:")
#     # pprint(detected_objects)

#     # Extract objects using LLM
#     extracted_objects = extract_objects(transcribed_text)
#     print("\nExtracted Objects:")
#     pprint(extracted_objects)

#     # Match detected objects with extracted objects
#     matched_objects = [
#         obj for obj in detected_objects
#         if obj['class_name'] in extracted_objects
#     ]

#     print("\nMatched Objects:")
#     pprint(matched_objects)

# if __name__ == "__main__":
#     main()