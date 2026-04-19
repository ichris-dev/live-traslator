import queue
import json
import time
import threading
import sys
import shutil
import sounddevice as sd
import ctranslate2
from vosk import Model, KaldiRecognizer
from transformers import AutoTokenizer



VOSK_MODEL_PATH = r"C:\Users\timot\speechProject\backend\combination\vosk-model-small-en-us-0.15\vosk-model-small-en-us-0.15"
CT2_MODEL_PATH = r"C:\Users\timot\speechProject\backend\translation_files\nllb-200-600M-ct2-int8"

SRC_LANG = "eng_Latn"
TGT_LANG = "kin_Latn"

CANDIDATE_RATES = [16000, 44100, 48000]
BLOCKSIZE = 4000

CT2_DEVICE = "cpu"
CT2_COMPUTE_TYPE = "int8"


audio_queue = queue.Queue()
translation_request_queue = queue.Queue()

# All confirmed Kinyarwanda words accumulated in one growing string
confirmed_kinyarwanda = ""
final_english_text = ""   # only used for the end-of-session summary (never printed live)

print_lock = threading.Lock()


vosk_model = Model(VOSK_MODEL_PATH)

translator = ctranslate2.Translator(
    CT2_MODEL_PATH,
    device=CT2_DEVICE,
    compute_type=CT2_COMPUTE_TYPE,
    intra_threads=4,
    inter_threads=2,
)

tokenizer = AutoTokenizer.from_pretrained(CT2_MODEL_PATH)
tokenizer.src_lang = SRC_LANG


def callback(indata, frames, time_info, status):
    if status:
        pass   
    audio_queue.put(bytes(indata))


def reset_state():
    global final_english_text, confirmed_kinyarwanda
    final_english_text = ""
    confirmed_kinyarwanda = ""

    for q in (audio_queue, translation_request_queue):
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
    device_names = ["Microphone Array", "Microphone", "Mic", "Realtek(R) Audio"]
    return find_working_device(device_names, CANDIDATE_RATES)


def get_system_audio_device():
    device_names = ["Stereo Mix", "Realtek HD Audio Stereo input"]
    return find_working_device(device_names, CANDIDATE_RATES)


def translate_chunk_local(text: str) -> str:
    text = text.strip()
    if not text:
        return ""

    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"][0].tolist()
    input_tokens = tokenizer.convert_ids_to_tokens(input_ids)

    results = translator.translate_batch(
        [input_tokens],
        target_prefix=[[TGT_LANG]],
        max_batch_size=1,
        beam_size=1,
    )

    output_tokens = results[0].hypotheses[0]
    if output_tokens and output_tokens[0] == TGT_LANG:
        output_tokens = output_tokens[1:]

    output_ids = tokenizer.convert_tokens_to_ids(output_tokens)
    return tokenizer.decode(output_ids, skip_special_tokens=True).strip()


def warm_up_local_translator():
    # Run silently — only print the result, no English labels
    translate_chunk_local("Hello my friend")


def render_line(confirmed: str, live: str = ""):
    """
    Always rewrite the SAME terminal line.
    Layout:  <confirmed words>  <live partial in dim>
    The tail of the combined string is shown so it always fits on one line.
    """
    cols = shutil.get_terminal_size((120, 20)).columns - 1

    if live:
        # Dim the live partial so the user knows it is still in progress
        display = confirmed + (" " if confirmed else "") + f"\033[2m{live}\033[0m"
        # For length calculation strip ANSI codes
        raw_len = len(confirmed) + (1 if confirmed else 0) + len(live)
    else:
        display = confirmed
        raw_len = len(confirmed)

    # If total length exceeds terminal width, keep only the rightmost tail
    if raw_len > cols:
        overflow = raw_len - cols
        if live:
            # Trim from the confirmed portion first
            trim = min(overflow, len(confirmed))
            confirmed_trimmed = confirmed[trim:]
            overflow -= trim
            live_trimmed = live[overflow:] if overflow > 0 else live
            display = confirmed_trimmed + (" " if confirmed_trimmed else "") + f"\033[2m{live_trimmed}\033[0m"
        else:
            display = confirmed[overflow:]

    sys.stdout.write("\r" + " " * cols + "\r")   # clear the line
    sys.stdout.write(display)
    sys.stdout.flush()



def translation_worker(stop_event: threading.Event):
    global confirmed_kinyarwanda

    while not stop_event.is_set():
        try:
            kind, english_chunk = translation_request_queue.get(timeout=0.2)
        except queue.Empty:
            continue

        if not english_chunk:
            continue

        try:
            translated = translate_chunk_local(english_chunk)
            if not translated:
                continue

            with print_lock:
                if kind == "final":
                    # Merge into the confirmed string — stays on the same line
                    confirmed_kinyarwanda = (confirmed_kinyarwanda + " " + translated).strip()
                    render_line(confirmed_kinyarwanda)          # no live partial anymore

                elif kind == "partial":
                    # Show confirmed + live partial — still one line
                    render_line(confirmed_kinyarwanda, live=translated)

        except Exception:
            pass   # swallow errors silently — never print English error text live


def run_recognition(device_name, samplerate, title):
    global final_english_text

    reset_state()
    recognizer = KaldiRecognizer(vosk_model, samplerate)
    stop_event = threading.Event()
    last_sent_partial = ""

    # Only status line uses English — printed once before the live line begins
    print(f"[{title} | {samplerate} Hz | Ctrl+C to stop]")
    print()   # blank line — the next sys.stdout.write will own this line

    def recognize_speech():
        global final_english_text
        nonlocal last_sent_partial

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
                    last_sent_partial = ""

                    if text:
                        final_english_text = (final_english_text + " " + text).strip()
                        translation_request_queue.put(("final", text))

                else:
                    partial = json.loads(recognizer.PartialResult())
                    partial_text = partial.get("partial", "").strip()

                    if partial_text and partial_text != last_sent_partial:
                        last_sent_partial = partial_text
                        translation_request_queue.put(("partial", partial_text))

    recognizer_thread  = threading.Thread(target=recognize_speech, daemon=True)
    translator_thread_1 = threading.Thread(target=translation_worker, args=(stop_event,), daemon=True)
    translator_thread_2 = threading.Thread(target=translation_worker, args=(stop_event,), daemon=True)

    translator_thread_1.start()
    translator_thread_2.start()
    recognizer_thread.start()

    try:
        while recognizer_thread.is_alive():
            time.sleep(0.1)
    except KeyboardInterrupt:
        stop_event.set()
        time.sleep(0.5)
        # Move off the live line before printing the summary
        print("\n")
        print("Final Kinyarwanda:", confirmed_kinyarwanda.strip())


try:
    print("Loading models...")
    warm_up_local_translator()
    print("Ready.\n")

    print_devices()
    mode = choose_mode()

    if mode == "1":
        device_name, samplerate = get_microphone_device()
        if samplerate is None:
            raise RuntimeError("Could not find a working microphone input device.")
        run_recognition(device_name, samplerate, "Microphone")

    elif mode == "2":
        device_name, samplerate = get_system_audio_device()
        if samplerate is None:
            raise RuntimeError(
                "Could not find a working system-audio input device. "
                "Make sure Stereo Mix is enabled in Windows."
            )
        run_recognition(device_name, samplerate, "System audio")

    else:
        print("Invalid choice. Run again and choose 1 or 2.")

except KeyboardInterrupt:
    print("\nStopped.")
except Exception as e:
    print("\nError:", e)