import os
import queue
import json
import time
import threading
import shutil
import sys
import sounddevice as sd
from vosk import Model, KaldiRecognizer

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.callbacks import BaseCallbackHandler


# =========================
# CONFIG
# =========================
MODEL_PATH = r"C:\Users\timot\speechProject\combination\vosk-model-small-en-us-0.15\vosk-model-small-en-us-0.15"
CANDIDATE_RATES = [16000, 44100, 48000]
BLOCKSIZE = 8000
LIVE_UPDATE_INTERVAL = 0.35

OPENROUTER_API_KEY = "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"

os.environ["OPENAI_API_KEY"] = OPENROUTER_API_KEY
os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"



audio_queue = queue.Queue()
translation_request_queue = queue.Queue()
display_queue = queue.Queue()

model = Model(MODEL_PATH)

final_text = ""
last_partial = ""
last_update_time = 0.0
translator_ready = False

class LiveTranslationHandler(BaseCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        display_queue.put(("token", token))

    def on_llm_end(self, response, **kwargs) -> None:
        display_queue.put(("done", None))



system_message = SystemMessage(
    content=(
        "You are a translation assistant. "
        "Translate the user's English text into Kinyarwanda only. "
        "Return only the translation, with no explanation."
    )
)

llm = ChatOpenAI(
    model="google/gemini-2.5-flash-lite",
    temperature=0,
    max_tokens=80,  # type: ignore
    streaming=True,
    callbacks=[LiveTranslationHandler()],
)



def callback(indata, frames, time_info, status):
    if status:
        print(status)
    audio_queue.put(bytes(indata))


def reset_state():
    global final_text, last_partial, last_update_time
    final_text = ""
    last_partial = ""
    last_update_time = 0.0

    for q in (audio_queue, translation_request_queue, display_queue):
        while not q.empty():
            try:
                q.get_nowait()
            except queue.Empty:
                break


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


def update_terminal_one_line(text):
    width = shutil.get_terminal_size((120, 20)).columns
    usable_width = max(20, width - 1)

    if len(text) > usable_width:
        text = text[-usable_width:]

    sys.stdout.write("\r" + " " * usable_width)
    sys.stdout.write("\r" + text.ljust(usable_width))
    sys.stdout.flush()


def warm_up_translator():
    global translator_ready
    try:
        response = llm.invoke([
            system_message,
            HumanMessage(content="hello")
        ])
        print(response)
        response2 = llm.invoke([
            system_message,
            HumanMessage(content="how are you")
        ])
        print(response2)
        translator_ready = True
        print("Translator ready.")
    except Exception as e:
        print(f"Translator warm-up failed: {e}")



def translation_worker(stop_event):
    while not stop_event.is_set():
        try:
            text_to_translate = translation_request_queue.get(timeout=0.2)
        except queue.Empty:
            continue

        if not text_to_translate:
            continue

        try:
            llm.invoke([
                system_message,
                HumanMessage(content=text_to_translate)
            ])
        except Exception as e:
            display_queue.put(("error", str(e)))


def display_worker(stop_event):
    full_translated_text = ""
    partial_stream = ""

    while not stop_event.is_set():
        try:
            item_type, value = display_queue.get(timeout=0.2)
        except queue.Empty:
            continue

        if item_type == "token":
            partial_stream += value
            current = (full_translated_text + " " + partial_stream).strip()
            update_terminal_one_line(current)

        elif item_type == "done":
            full_translated_text = (full_translated_text + " " + partial_stream).strip()
            partial_stream = ""
            update_terminal_one_line(full_translated_text)

        elif item_type == "error":
            update_terminal_one_line(f"Ikosa: {value}")

    return full_translated_text



def run_recognition(device_name, samplerate, title):
    global final_text, last_partial, last_update_time

    reset_state()
    recognizer = KaldiRecognizer(model, samplerate)
    stop_event = threading.Event()

    translated_store = {
        "full_text": ""
    }

    print(f"\nUsing device: {device_name if device_name is not None else 'Default microphone'}")
    print(f"Using samplerate: {samplerate}")
    print(f"{title}... Press Ctrl+C to stop.\n")

    def display_worker_with_store():
        full_translated_text = ""
        partial_stream = ""

        while not stop_event.is_set():
            try:
                item_type, value = display_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if item_type == "token":
                partial_stream += value
                current = (full_translated_text + " " + partial_stream).strip()
                update_terminal_one_line(current)

            elif item_type == "done":
                full_translated_text = (full_translated_text + " " + partial_stream).strip()
                translated_store["full_text"] = full_translated_text
                partial_stream = ""
                update_terminal_one_line(full_translated_text)

            elif item_type == "error":
                update_terminal_one_line(f"Ikosa: {value}")

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
                try:
                    data = audio_queue.get(timeout=0.2)
                except queue.Empty:
                    continue

                if recognizer.AcceptWaveform(data):
                    result = json.loads(recognizer.Result())
                    text = result.get("text", "").strip()

                    if text:
                        final_text = (final_text + " " + text).strip()
                        last_partial = ""
                        translation_request_queue.put(text)

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
    

    recognizer_thread = threading.Thread(target=recognize_speech, daemon=True)
    translator_thread = threading.Thread(target=translation_worker, args=(stop_event,), daemon=True)
    printer_thread = threading.Thread(target=display_worker_with_store, daemon=True)

    translator_thread.start()
    printer_thread.start()
    recognizer_thread.start()

    try:
        while recognizer_thread.is_alive():
            time.sleep(0.1)
    except KeyboardInterrupt:
        stop_event.set()
        time.sleep(0.3)
        print("\n\nStopped.")
        print("Final English:", final_text.strip())
        print("Final Kinyarwanda:", translated_store["full_text"].strip())



try:
    print("Warming up translator...")
    warm_up_translator()

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