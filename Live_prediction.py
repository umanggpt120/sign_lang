import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks import python
import numpy as np
import tensorflow as tf
from collections import deque
import threading
import queue
import subprocess
import os
import tempfile

from deep_translator import GoogleTranslator
from gtts import gTTS
import pygame

pygame.mixer.init()

language_mode = "en"
language_lock = threading.Lock()

ISL_EXPAND_MAP = {
    "i water"         : "I need water.",
    "i food"          : "I need food.",
    "i go"            : "I want to go.",
    "i go school"     : "I am going to school.",
    "i go home"       : "I want to go home.",
    "i go hospital"   : "I need to go to the hospital.",
    "i sick"          : "I am feeling sick.",
    "i hungry"        : "I am hungry.",
    "i thirsty"       : "I am thirsty.",
    "i tired"         : "I am tired.",
    "i happy"         : "I am happy.",
    "i sad"           : "I am sad.",
    "i pain"          : "I am in pain.",
    "i help"          : "I need help.",
    "you help me"     : "Can you help me?",
    "i toilet"        : "I need to use the toilet.",
    "i medicine"      : "I need medicine.",
    "i sleep"         : "I want to sleep.",
    "i sit"           : "I want to sit.",
    "i stand"         : "I want to stand.",
    "i name"          : "What is your name?",
    "thank you"       : "Thank you.",
    "sorry"           : "I am sorry.",
    "i understand"    : "I understand.",
    "i not understand": "I do not understand.",
    "please repeat"   : "Please repeat that.",
    "i call"          : "I need to make a call.",
    "i mother"        : "I want my mother.",
    "i father"        : "I want my father.",
    "i doctor"        : "I need a doctor.",
    "i police"        : "I need the police.",
    "i money"         : "I need money.",
    "i work"          : "I need to go to work.",
    "i learn"         : "I want to learn.",
    "i write"         : "I want to write.",
    "i read"          : "I want to read.",
    "i phone"         : "I need my phone.",
    "i cold"          : "I am feeling cold.",
    "i hot"           : "I am feeling hot.",
    "i love you"      : "I love you.",
    "i like you"      : "I like you.",
    "i miss you"      : "I miss you.",
    "i fine"          : "I am fine.",
    "i okay"          : "I am okay.",
    "i danger"        : "I am in danger.",
    "i lost"          : "I am lost.",
    "i emergency"     : "This is an emergency.",
    "help me"         : "Please help me.",
    "i angry"         : "I am angry.",
    "i afraid"        : "I am afraid.",
    "i alone"         : "I am alone.",
    "i deaf"          : "I am deaf.",
    "i mute"          : "I am mute.",
}

def build_expanded_input(raw_words):
    raw = " ".join(raw_words).lower().strip()
    if raw in ISL_EXPAND_MAP:
        return ISL_EXPAND_MAP[raw]
    for key, val in ISL_EXPAND_MAP.items():
        if raw.startswith(key):
            return val
    if len(raw_words) == 1:
        return f"I need {raw_words[0].lower()}."
    if raw_words[0].lower() == "i" and len(raw_words) == 2:
        return f"I need {raw_words[1]}."
    if raw_words[0].lower() == "i" and len(raw_words) >= 3:
        return f"I want to {' '.join(raw_words[1:])}."
    return raw.capitalize() + "."

def expand_isl_sentence(raw_words):
    try:
        result = build_expanded_input(raw_words)
        print(f" Expanded (EN): {result}")
        return result
    except Exception as e:
        print(f" Expansion failed: {e}")
        return " ".join(raw_words).capitalize() + "."

def translate_to_hindi(text):
    try:
        translated = GoogleTranslator(source='en', target='hi').translate(text.strip())
        if translated and translated.strip():
            print(f"🇮🇳 Hindi: {translated}")
            return translated.strip()
        print(" Empty translation — using English fallback.")
        return text
    except Exception as e:
        print(f" Translation error: {e} — using English fallback.")
        return text

speech_queue = queue.Queue()

def _speak_english(text):
    subprocess.run(
        ["powershell", "-Command",
         f'Add-Type -AssemblyName System.Speech; '
         f'$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; '
         f'$s.Rate = -3; '
         f'$s.Speak("{text}");'],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

def _speak_hindi(text):
    tmp_path = None
    try:
        tts = gTTS(text=text, lang='hi', slow=False)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as f:
            tts.save(f.name)
            tmp_path = f.name
        pygame.mixer.music.load(tmp_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
    except Exception as e:
        print(f" Hindi TTS error: {e}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                pygame.mixer.music.unload()
                os.unlink(tmp_path)
            except Exception:
                pass

def speech_worker():
    while True:
        item = speech_queue.get()
        if item is None:
            break
        english_text, mode = item
        print(f" Speaking [{mode.upper()}]: {english_text}")
        if mode == "hi":
            hindi_text = translate_to_hindi(english_text)
            _speak_hindi(hindi_text)
        else:
            _speak_english(english_text)
        speech_queue.task_done()

speech_thread = threading.Thread(target=speech_worker, daemon=True)
speech_thread.start()

def speak(english_text):
    while not speech_queue.empty():
        try:
            speech_queue.get_nowait()
            speech_queue.task_done()
        except Exception:
            pass
    with language_lock:
        mode = language_mode
    speech_queue.put((english_text, mode))

SEQ_LEN              = 30
CONFIDENCE_THRESHOLD = 0.80
PRED_HISTORY_LEN     = 10
PRED_INTERVAL        = 5

MODEL_PATH  = "gesture_lstm_model.h5"
LABELS_PATH = "label_classes.npy"


model  = tf.keras.models.load_model(MODEL_PATH)
labels = list(np.load(LABELS_PATH, allow_pickle=True))
print(f" Model loaded. Classes: {labels}")

def normalize_landmarks(sequence):
    seq = sequence.copy()
    for i in range(seq.shape[0]):
        wrist1 = seq[i, 0:3].copy()
        wrist2 = seq[i, 63:66].copy()
        seq[i, :63] -= np.tile(wrist1, 21)
        seq[i, 63:]  -= np.tile(wrist2, 21)
    return seq

def scale_normalize_sequence(seq):
    result = seq.copy()
    for i in range(seq.shape[0]):
        frame    = seq[i].reshape(-1, 3)
        max_dist = np.max(np.linalg.norm(frame, axis=1))
        if max_dist > 1e-6:
            result[i] = seq[i] / max_dist
    return result

base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_hands=2
)
landmarker = vision.HandLandmarker.create_from_options(options)

def extract_frame_landmarks(result, frame, w, h, draw=True):
    if not result.hand_landmarks:
        return None
    frame_data = []
    for hand_id in range(2):
        if hand_id < len(result.hand_landmarks):
            hand = result.hand_landmarks[hand_id]
            for lm in hand:
                frame_data.extend([lm.x, lm.y, lm.z])
                if draw:
                    cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 4, (0, 255, 0), -1)
        else:
            frame_data.extend([0.0] * 63)
    return frame_data


def draw_prediction(frame, text, conf, color=(0, 0, 255)):
    cv2.putText(frame, f"{text} ({conf:.2f})", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

def draw_buffer_bar(frame, filled, total, w):
    bar_w = int((filled / total) * (w - 40))
    cv2.rectangle(frame, (20, 70), (w - 20, 90), (50, 50, 50), -1)
    cv2.rectangle(frame, (20, 70), (20 + bar_w, 90), (0, 200, 100), -1)
    cv2.putText(frame, f"Buffer: {filled}/{total}", (20, 108),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

def draw_sentence(frame, words, expanded, hindi, w):
    """Display Raw / EN / HI lines with a dark background strip."""
    raw     = " ".join(words) if words else "(no gestures yet)"
    en_disp = expanded if expanded else "..."
    hi_disp = hindi    if hindi    else "..."


    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 118), (w - 10, 215), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    cv2.putText(frame, f"Raw : {raw}",
                (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255,   0), 2)
    cv2.putText(frame, f"EN  : {en_disp}",
                (20, 168), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0,   255, 255), 2)
    

def draw_hud(frame, mode):
    """Bottom bar: active language mode + keyboard shortcut hints."""
    h, w = frame.shape[:2]

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 36), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    mode_text  = "[ EN - English ]" if mode == "en" else "[ HI - Hindi ]"
    mode_color = (100, 220, 255)     if mode == "en" else (100, 255, 160)
    cv2.putText(frame, mode_text, (20, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, mode_color, 1)

    hint = "S=Speak   C=Clear   D=Delete last   H=Hindi   E=English   Q=Quit"
    cv2.putText(frame, hint, (210, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (170, 170, 170), 1)

sentence_words    = []
expanded_sentence = ""
hindi_sentence    = ""
expansion_lock    = threading.Lock()

def expand_and_translate_bg(words_snapshot):
    global expanded_sentence, hindi_sentence
    if not words_snapshot:
        with expansion_lock:
            expanded_sentence = ""
            hindi_sentence    = ""
        return
    en = expand_isl_sentence(words_snapshot)
    hi = translate_to_hindi(en)
    with expansion_lock:
        expanded_sentence = en
        hindi_sentence    = hi

def _startup_test():
    test_en = "I am happy."
    test_hi = translate_to_hindi(test_en)
    if test_hi != test_en:
        print(f" Translation OK: '{test_en}' → '{test_hi}'")
    else:
        print(" Translation NOT working — check internet & pip install deep-translator")
threading.Thread(target=_startup_test, daemon=True).start()

cap          = cv2.VideoCapture(0)
sequence     = deque(maxlen=SEQ_LEN)
pred_history = deque(maxlen=PRED_HISTORY_LEN)

current_label    = ""
current_conf     = 0.0
frame_counter    = 0
last_added_label = ""

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    frame_counter += 1

    rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result   = landmarker.detect(mp_image)

    frame_data = extract_frame_landmarks(result, frame, w, h, draw=True)

    if frame_data is not None:
        sequence.append(frame_data)
    else:
        sequence.clear()
        pred_history.clear()
        current_label = ""
        current_conf  = 0.0

    if len(sequence) == SEQ_LEN and frame_counter % PRED_INTERVAL == 0:
        seq_np = np.array(sequence, dtype=np.float32)
        seq_np = normalize_landmarks(seq_np)
        seq_np = scale_normalize_sequence(seq_np)

        pred      = model(np.expand_dims(seq_np, axis=0), training=False).numpy()[0]
        conf      = float(np.max(pred))
        label_idx = int(np.argmax(pred))

        if conf >= CONFIDENCE_THRESHOLD:
            pred_history.append(label_idx)

        if len(pred_history) >= 5:
            voted_idx     = max(set(pred_history), key=pred_history.count)
            new_label     = labels[voted_idx]
            current_label = new_label
            current_conf  = conf

            if new_label != last_added_label:
                sentence_words.append(new_label)
                last_added_label = new_label
                print(f" Added: {new_label} → {sentence_words}")
                threading.Thread(
                    target=expand_and_translate_bg,
                    args=(list(sentence_words),),
                    daemon=True
                ).start()

        elif len(pred_history) == 0:
            current_label = "..."
            current_conf  = 0.0

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    elif key == ord('s'):                        
        if sentence_words:
            with expansion_lock:
                to_speak = expanded_sentence if expanded_sentence else " ".join(sentence_words)
            print(f" [S] Speaking: {to_speak}")
            speak(to_speak)
        else:
            speak("No gestures recorded.")

    elif key == ord('c'):                        
        sentence_words.clear()
        with expansion_lock:
            expanded_sentence = ""
            hindi_sentence    = ""
        last_added_label = ""
        print(" [C] Cleared.")

    elif key == ord('d'):                        
        if sentence_words:
            removed = sentence_words.pop()
            print(f" [D] Removed: {removed}")
            threading.Thread(
                target=expand_and_translate_bg,
                args=(list(sentence_words),),
                daemon=True
            ).start()

    elif key == ord('h'):                        
        with language_lock:
            language_mode = "hi"
        print(" [H] Language: Hindi")

    elif key == ord('e'):                        
        with language_lock:
            language_mode = "en"
        print(" [E] Language: English")

    if current_label:
        color = (0, 200, 0) if current_conf >= CONFIDENCE_THRESHOLD else (0, 165, 255)
        draw_prediction(frame, current_label, current_conf, color)
    else:
        cv2.putText(frame, "Detecting...", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (180, 180, 180), 2)

    draw_buffer_bar(frame, len(sequence), SEQ_LEN, w)

    with expansion_lock:
        en_snap = expanded_sentence
        hi_snap = hindi_sentence

    with language_lock:
        cur_mode = language_mode

    draw_sentence(frame, sentence_words, en_snap, hi_snap, w)
    draw_hud(frame, cur_mode)

    cv2.imshow("ISL Gesture Recognition", frame)

speech_queue.put(None)
cap.release()
cv2.destroyAllWindows()
