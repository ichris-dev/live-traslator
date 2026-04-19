# Importing necessary libraries
import queue
import json
import time
import sounddevice as sd
import threading
from vosk import Model, KaldiRecognizer

# Defining the model path in the variable MODEL_PATH
MODEL_PATH = r"C:\Users\timot\speechProject\combination\vosk-model-small-en-us-0.15\vosk-model-small-en-us-0.15"
CANDIDATE_RATES = [16000, 44100, 48000]
BLOCKSIZE = 8000
LIVE_UPDATE_INTERVAL = 0.8

q = queue.Queue()
text_queue = queue.Queue()
model = Model(MODEL_PATH)

final_text = ""
last_partial = ""
last_update_time = 0.0


def callback(indata, frames, time_info, status):
    if status:
        print(status)
    q.put(bytes(indata))


def reset_state():
    global final_text, last_partial, last_update_time
    final_text = ""
    last_partial = ""
    last_update_time = 0.0


def print_devices():
    print("\nAvailable audio devices:\n")
    print(sd.query_devices())
    print()


def choose_mode():
    print("Choose input source:")
    print("1. Normal speech (microphone)")
    print("2. YouTube / system audio")
    return input("Enter 1 or 2: ").strip()


def find_working_device(device_names, prefer_rates):
    for device_name in device_names:
        for rate in prefer_rates:
            try:
                sd.check_input_settings(
                    device=device_name,
                    channels=1,
                    dtype="int16",
                    samplerate=rate
                )
                return device_name, rate
            except Exception:
                pass
    return None, None


def get_microphone_device():
    for rate in CANDIDATE_RATES:
        try:
            sd.check_input_settings(
                device=None,
                channels=1,
                dtype="int16",
                samplerate=rate
            )
            return None, rate
        except Exception:
            pass

    device_names = [
        "Microphone Array",
        "Microphone",
        "Mic",
        "Realtek(R) Audio"
    ]
    return find_working_device(device_names, CANDIDATE_RATES)


def get_system_audio_device():
    device_names = [
        "Stereo Mix",
        "Realtek HD Audio Stereo input"
    ]
    return find_working_device(device_names, CANDIDATE_RATES)

def run_recognition(device_name, samplerate, title):
    global final_text, last_partial, last_update_time

    import threading
    import shutil
    import sys

    reset_state()
    recognizer = KaldiRecognizer(model, samplerate)
    stop_event = threading.Event()

    print(f"\nUsing device: {device_name if device_name is not None else 'Default microphone'}")
    print(f"Using samplerate: {samplerate}")
    print(f"{title}... Press Ctrl+C to stop.\n")

    def one_line(text):
        width = shutil.get_terminal_size((120, 20)).columns
        safe_width = max(10, width - 1)

        if len(text) > safe_width:
            text = text[-safe_width:]

        sys.stdout.write("\r" + " " * safe_width)
        sys.stdout.write("\r" + text.ljust(safe_width))
        sys.stdout.flush()

    def recognize_speech():
        global final_text, last_partial, last_update_time

        with sd.RawInputStream(
            samplerate=samplerate,
            blocksize=BLOCKSIZE,
            dtype="int16",
            channels=1,
            callback=callback,
            device=device_name,
        ):
            while not stop_event.is_set():
                data = q.get()

                if recognizer.AcceptWaveform(data):
                    result = json.loads(recognizer.Result())
                    text = result.get("text", "").strip()

                    if text:
                        final_text = (final_text + " " + text).strip()
                        last_partial = ""
                        text_queue.put(final_text)

                else:
                    partial = json.loads(recognizer.PartialResult())
                    partial_text = partial.get("partial", "").strip()
                    now = time.time()

                    if (
                        partial_text
                        and partial_text != last_partial
                        and (now - last_update_time) >= LIVE_UPDATE_INTERVAL
                    ):
                        last_partial = partial_text
                        last_update_time = now
                        live_line = (final_text + " " + partial_text).strip()
                        text_queue.put(live_line)

    def print_recognized():
        current_text = ""
        while not stop_event.is_set():
            new_text = text_queue.get()
            if new_text != current_text:
                current_text = new_text
                one_line(current_text)

    recognizer_thread = threading.Thread(target=recognize_speech, daemon=True)
    printer_thread = threading.Thread(target=print_recognized, daemon=True)

    recognizer_thread.start()
    printer_thread.start()

    try:
        while recognizer_thread.is_alive():
            time.sleep(0.1)
    except KeyboardInterrupt:
        stop_event.set()
        print("\n\nStopped.")
        print("Final text:", final_text.strip())


try:
    print_devices()
    mode = choose_mode()

    if mode == "1":
        device_name, samplerate = get_microphone_device()
        if samplerate is None:
            raise RuntimeError("Could not find a working microphone input device.")
        run_recognition(device_name, samplerate, "Listening to microphone")

    elif mode == "2":
        device_name, samplerate = get_system_audio_device()
        if samplerate is None:
            raise RuntimeError(
                "Could not find a working system-audio input device. "
                "Make sure Stereo Mix is enabled in Windows."
            )
        run_recognition(device_name, samplerate, "Listening to YouTube/system audio")

    else:
        print("Invalid choice. Run again and choose 1 or 2.")

except KeyboardInterrupt:
    print("\n\nStopped.")
    print("Final text:", final_text.strip())
except Exception as e:
    print("\nError:", e)