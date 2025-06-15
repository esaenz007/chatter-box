import pygame
import sys
import threading
import queue
import random
import pronouncing
import speech_recognition as sr
import sounddevice as sd
import scipy.io.wavfile as wavfile
import numpy as np
import os
import pyttsx3
import json
import openai
import vosk
import math
import logging
import subprocess
from collections import deque

class OldTV:

    def __init__(self):
        try:
            self.debug_mode = False
            self.show_settings = False
            self.settings_voice_index = 0
            self.settings_selected_voice = None
            self.settings_voice_rate = 200  # Default rate, adjust as needed
            self.settings_voice_list = []
            self.settings_rate_min = 80
            self.settings_rate_max = 300
            self.settings_rate_step = 10
            self.settings_selected_recording = None
            self.settings_edit_text = ""

            self.log_buffer = deque(maxlen=15)  # Show last 15 log lines

            # Set up logging to buffer and console
            class UILogHandler(logging.Handler):
                def __init__(self, buffer):
                    super().__init__()
                    self.buffer = buffer
                def emit(self, record):
                    msg = self.format(record)
                    self.buffer.append(msg)

            self.logger = logging.getLogger("OldTV")
            self.logger.setLevel(logging.DEBUG)
            handler = UILogHandler(self.log_buffer)
            formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
            handler.setFormatter(formatter)
            self.logger.handlers = [handler]

            # Add console handler
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

            self.SAVE_FILE = "recordings.json"
            self.recognizer = sr.Recognizer()
            self.sample_rate = 44100
            self.recordings = {}
            self.load_recordings()
            self.recording = False
            self.awaiting_save = False
            self.audio_data = []
            self.last_transcribed_text = ""
            self.neutral_face = ":)"
            self.speaking_face = ":D"
            self.current_face = self.neutral_face
            self.face_queue = queue.Queue()
            self.current_mouth = ")"
            self._VOWELS = set("aeiouy")
            self.PHONEME_TO_FACE = {
                'AA': 'D', 'AE': 'D', 'AH': 'D', 'AO': 'O', 'AW': 'D', 'AY': 'D',
                'B': 'D', 'CH': 'D', 'D': 'D', 'DH': 'D', 'EH': 'D', 'ER': 'D',
                'EY': 'D', 'F': 'D', 'G': 'D', 'HH': 'D', 'IH': 'D', 'IY': 'D',
                'JH': 'D', 'K': 'D', 'L': 'D', 'M': '|', 'N': 'D', 'NG': 'D',
                'OW': 'D', 'OY': 'D', 'P': '|', 'R': 'D', 'S': 'D', 'SH': 'D',
                'T': 'D', 'TH': 'D', 'UH': 'D', 'UW': 'O', 'V': 'D', 'W': 'D',
                'Y': 'D', 'Z': 'D', 'ZH': 'D'
            }
            self.syllable_count = 0
            self.SCREEN_WIDTH = 640
            self.SCREEN_HEIGHT = 480
            self.NOISE_DOTS = 500
            self.FONT_SIZE = int(self.SCREEN_HEIGHT * 0.75)
            self.wake_word_thread = None
            self.wake_word_stop_event = threading.Event()
            self.logger.info("Initializing Pygame...")
            try:
                pygame.init()
                self.logger.info("Pygame initialized successfully")
            except Exception as e:
                self.logger.info(f"Pygame initialization failed: {e}")
                sys.exit(1)
            pygame.rand = random
            try:
                self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
                pygame.display.set_caption("Old TV Emoji Faces")
                self.logger.info("Pygame display set")
            except Exception as e:
                self.logger.info(f"Pygame display setup failed: {e}")
                sys.exit(1)
            self.clock = pygame.time.Clock()
            font_name = pygame.font.match_font("dejavusans")
            if font_name:
                self.font = pygame.font.Font(font_name, self.FONT_SIZE)
                self.logger.info(f"Font loaded: {font_name}")
            else:
                self.logger.info("Font 'dejavusans' not found")
                sys.exit(1)
            self.interference_surface = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            self.logger.info("Initializing TTS engine...")
            self.interrupt_tts = False
            self.face_queue = queue.Queue()
            try:
                self.engine = pyttsx3.init()
                self.engine.connect("finished-utterance", self.on_end)
                self.engine.connect("started-utterance", self.on_start)
                self.engine.connect("started-word", self.on_start_word)
                self.engine.connect("error", self.on_error)
                self.logger.info("TTS engine initialized")
                # Settings: get voices and rate
                self.settings_voice_list = self.engine.getProperty("voices")
                self.settings_voice_index = 0
                self.settings_selected_voice = self.engine.getProperty("voice")
                self.settings_voice_rate = self.engine.getProperty("rate")
            except Exception as e:
                self.logger.info(f"TTS initialization failed: {e}")
                sys.exit(1)
            self.is_talking = False
            self.current_index = 0
            self.FACE_WAIT = ")"
            self.FACE_LOUD = "D"
            self.FACE_NORMAL = "D"
            self.FACE_QUIET = "D"
            self.FACE_IDLE = ")"
            self.next_phrase = ""
            self.listening_state = "wake_word"
            openai_key = os.environ.get("OPENAI_API_KEY")
            self.ai_client = openai.OpenAI(api_key=openai_key)

            self.is_pi = self.is_raspberry_pi()
            self.undervoltage_warning = False
            self.pi_voltage = "Unavailable"
        except Exception as e:
            print(f"Error in __init__: {e}")

    def stop_wake_word_listener(self):
        try:
            self.wake_word_stop_event.set()
            if (
                self.wake_word_thread
                and self.wake_word_thread.is_alive()
                and threading.current_thread() != self.wake_word_thread
            ):
                self.wake_word_thread.join(timeout=2)
            self.wake_word_thread = None
        except Exception as e:
            self.logger.error(f"Error in stop_wake_word_listener: {e}")

    def run_wake_word_listener(self):
        try:
            self.wake_word_stop_event.clear()
            self.wake_word_thread = threading.Thread(target=self.start_wake_word_listener, daemon=True)
            self.wake_word_thread.start()
        except Exception as e:
            self.logger.error(f"Error in run_wake_word_listener: {e}")

    def record_audio(self):
        try:
            self.audio_data = []
            stream = None
            try:
                stream = sd.InputStream(samplerate=self.sample_rate, channels=1, dtype='int16')
                stream.start()
                self.logger.info("Recording started")
                while self.recording:
                    data = stream.read(1024)[0]
                    self.audio_data.append(data)
                stream.stop()
            except Exception as e:
                self.logger.info(f"Error during recording: {e}")
            finally:
                if stream is not None:
                    try:
                        stream.close()
                    except Exception:
                        pass
                if self.audio_data:
                    self.audio_data = np.concatenate(self.audio_data, axis=0)
                    wavfile.write("temp.wav", self.sample_rate, self.audio_data)
                    self.logger.info("Audio recorded to temp.wav")
                else:
                    self.logger.info("No audio data recorded")
        except Exception as e:
            self.logger.error(f"Error in record_audio: {e}")

    def speech_to_text(self):
        try:
            try:
                if not os.path.exists("temp.wav"):
                    self.logger.info("No audio file created")
                    return "No audio file created"
                with sr.AudioFile("temp.wav") as source:
                    audio = self.recognizer.record(source)
                    self.logger.info("Audio file read, sending to Google")
                    text = self.recognizer.recognize_google(audio, language="en-US")
                    self.logger.info(f"Recognized text: {text}")
                    return text
            except sr.UnknownValueError:
                self.logger.info("Speech recognition failed: Could not understand audio")
                return ""
            except sr.RequestError as e:
                self.logger.info(f"Speech recognition service error: {e}")
                return ""
            except Exception as e:
                self.logger.info(f"Error processing audio: {e}")
                return ""
            finally:
                if os.path.exists("temp.wav"):
                    try:
                        os.remove("temp.wav")
                        self.logger.info("Cleaned up temp.wav")
                    except Exception as e:
                        self.logger.info(f"Error removing temp.wav: {e}")
        except Exception as e:
            self.logger.error(f"Error in speech_to_text: {e}")

    def play_message(self, text):
        try:
            if not text:
                self.logger.info("No valid text to play")
                return False
            try:
                self.stop_wake_word_listener()
                self.next_phrase = text
                pygame.event.pump()
                def speech_thread():
                    try:
                        self.logger.info(f"Playing TTS: {text}")
                        if self.engine.isBusy():
                            self.engine.stop()
                        self.engine.say(text, "Message Playback")
                        self.engine.runAndWait()
                        self.logger.info("TTS playback completed")
                    except Exception as e:
                        self.logger.info(f"Error in TTS playback: {e}")
                    finally:
                        self.run_wake_word_listener()
                thread = threading.Thread(target=speech_thread)
                thread.start()
                # Do NOT join here; let the thread run in the background
                return True
            except Exception as e:
                self.logger.info(f"Error in play_message: {e}")
                self.run_wake_word_listener()
                return False
        except Exception as e:
            self.logger.error(f"Error in play_message (outer): {e}")

    def get_syllable_count(self, word: str) -> int:
        try:
            phones = pronouncing.phones_for_word(word.lower().strip(".,!?;:"))
            if not phones:
                return self.fallback_syllable_count(word)
            syllable_count = pronouncing.syllable_count(phones[0])
        except Exception as e:
            self.logger.info(f"Error counting syllables for '{word}': {e}")
            syllable_count = self.fallback_syllable_count(word)
        return syllable_count if syllable_count > 0 else 1

    def fallback_syllable_count(self, word: str) -> int:
        try:
            w = word.lower().rstrip(".,!?;:")
            if w.endswith("e"):
                w = w[:-1]
            count = 0
            prev_is_vowel = False
            vset = self._VOWELS
            for ch in w:
                is_vowel = ch in vset
                if is_vowel and not prev_is_vowel:
                    count += 1
                prev_is_vowel = is_vowel
            return count or 1
        except Exception as e:
            self.logger.info(f"Error in fallback_syllable_count: {e}")
            return 1

    def on_start_word(self, name, location, length):
        try:
            self.logger.info(f"Started word: {name}, Location: {location}, Length: {length}")
            self.is_talking = True
            if self.interrupt_tts:
                self.logger.info("Interrupting TTS playback")
                self.engine.stop()
                self.interrupt_tts = False
                self.is_talking = False
        except Exception as e:
            self.logger.info(f"Error in on_start_word: {e}")

    def on_start(self, name):
        try:
            self.logger.info(f"Started speaking: {name}")
            self.is_talking = True
        except Exception as e:
            self.logger.info(f"Error in on_start: {e}")

    def on_end(self, name, completed):
        try:
            self.logger.info(f"Finished speaking: {name}, Completed: {completed}")
            self.is_talking = False
        except Exception as e:
            self.logger.info(f"Error in on_end: {e}")
        
    def on_error(self, name, error):
        try:
            self.logger.info(f"Error in TTS: {name}, Error: {error}")
            self.is_talking = False
        except Exception as e:
            self.logger.info(f"Error in on_error: {e}")

    def draw_background(self):
        try:
            noise_speed = 5.0  # 1.0 = normal, <1.0 = slower, >1.0 = faster
            t = pygame.time.get_ticks() / 1000.0  # seconds
            frame = int(t * noise_speed)
            random.seed(frame)
            self.screen.fill((0, 40, 0))  # Dark CRT green
            for _ in range(self.NOISE_DOTS):
                x = random.randint(0, self.SCREEN_WIDTH)
                y = random.randint(0, self.SCREEN_HEIGHT)
                color = (0, random.randint(100, 255), 0)
                self.screen.set_at((x, y), color)
        except Exception as e:
            self.logger.info(f"Error drawing background: {e}")

    def draw_face(self, mouth=")"):
        try:
            wpm = 120  # Animation speed: words per minute
            wps = wpm / 60.0
            cycle_duration = 1.0 / wps  # seconds per word (open+close)
            half_cycle = cycle_duration / 2

            if self.is_talking:
                t = pygame.time.get_ticks() / 1000.0  # seconds
                phase = int((t % cycle_duration) // half_cycle)
                mouth_anim = self.FACE_NORMAL if phase == 0 else self.FACE_WAIT
            else:
                mouth_anim = self.FACE_IDLE

            eyes_surface = self.font.render(":", True, (0, 255, 0))
            mouth_surface = self.font.render(mouth_anim, True, (0, 255, 0))
            eyes_x = self.SCREEN_WIDTH // 2 - eyes_surface.get_width() - 20
            eyes_y = self.SCREEN_HEIGHT // 2 - eyes_surface.get_height() // 2
            mouth_x = eyes_x + eyes_surface.get_width() + 5
            mouth_y = eyes_y
            self.screen.blit(eyes_surface, (eyes_x, eyes_y))
            self.screen.blit(mouth_surface, (mouth_x, mouth_y))
        except Exception as e:
            self.logger.info(f"Error drawing face: {e}")

    def draw_interference(self):
        try:
            interference_speed = 5.0  # 1.0 = normal, <1.0 = slower, >1.0 = faster
            t = pygame.time.get_ticks() / 1000.0  # seconds
            frame = int(t * interference_speed)
            random.seed(1000 + frame)
            self.interference_surface.fill((0, 0, 0, 0))
            for _ in range(random.randint(1, 3)):
                y = random.randint(0, self.SCREEN_HEIGHT)
                height = random.randint(1, 3)
                alpha = random.randint(30, 100)
                color = (0, 255, 0, alpha)
                pygame.draw.rect(self.interference_surface, color, (0, y, self.SCREEN_WIDTH, height))
            self.screen.blit(self.interference_surface, (0, 0))
        except Exception as e:
            self.logger.info(f"Error drawing interference: {e}")

    def load_recordings(self):
        try:
            if os.path.exists(self.SAVE_FILE):
                try:
                    with open(self.SAVE_FILE, "r") as f:
                        self.recordings = json.load(f)
                    self.logger.info(f"Loaded recordings from {self.SAVE_FILE}")
                except Exception as e:
                    self.logger.info(f"Error loading recordings: {e}")
                    self.recordings = {}
            else:
                self.recordings = {}
        except Exception as e:
            self.logger.info(f"Error in load_recordings: {e}")

    def save_recordings(self):
        try:
            with open(self.SAVE_FILE, "w") as f:
                json.dump(self.recordings, f)
            self.logger.info(f"Saved recordings to {self.SAVE_FILE}")
        except Exception as e:
            self.logger.info(f"Error saving recordings: {e}")

    def beep(self, freq=440, duration=180, volume=0.3):
        try:
            pygame.mixer.init(frequency=22050, size=-16)
            sample_rate = 22050
            n_samples = int(round(duration * sample_rate / 1000))
            buf = (volume * 32767 * np.sin(2.0 * np.pi * np.arange(n_samples) * freq / sample_rate)).astype(np.int16)
            mixer_channels = pygame.mixer.get_init()[2]
            if mixer_channels == 1:
                if buf.ndim == 1:
                    arr = buf
                else:
                    arr = buf.reshape(-1)
            elif mixer_channels == 2:
                arr = np.repeat(buf[:, np.newaxis], 2, axis=1)
            else:
                raise ValueError(f"Unsupported mixer channels: {mixer_channels}")
            sound = pygame.sndarray.make_sound(arr)
            sound.play()
            pygame.time.delay(duration)
            sound.stop()
            pygame.mixer.quit()
        except Exception as e:
            self.logger.info(f"Beep error: {e}")

    def start_wake_word_listener(self):
        try:
            model_path = "models/vosk-model-small-en-us-0.15"
            if not os.path.exists(model_path):
                raise RuntimeError(f"Vosk model not found at {model_path}.  Current path: {os.getcwd()}")
            model = vosk.Model(model_path)
            recognizer = vosk.KaldiRecognizer(model, 16000)
            wake_words = ["hey tv", "computer", "stop"]

            q = queue.Queue()

            def audio_callback(indata, frames, time, status):
                if status:
                    self.logger.info(status)
                q.put(bytes(indata))

            while not self.wake_word_stop_event.is_set():
                self.listening_state = "wake_word"
                with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16',
                                       channels=1, callback=audio_callback):
                    self.logger.info("Listening for wake word ('hey tv', 'computer', or 'stop')...")
                    while not self.wake_word_stop_event.is_set():
                        data = q.get()
                        if recognizer.AcceptWaveform(data):
                            result = recognizer.Result()
                            import json
                            text = json.loads(result).get("text", "").lower()
                            self.logger.info(f"Recognized: {text}")
                            if "stop" in text:
                                self.logger.info("Wake word 'stop' detected!")
                                if self.engine.isBusy():
                                    self.logger.info("TTS engine is busy. Stopping TTS.")
                                    self.engine.stop()
                                continue
                            if any(word in text for word in ["hey tv", "computer"]):
                                self.logger.info("Wake word detected!")
                                break
                if not self.wake_word_stop_event.is_set():
                    self.handle_question()
                self.listening_state = None
        except Exception as e:
            self.logger.error(f"Error in start_wake_word_listener: {e}")

    def preprocess_for_tts(self, text):
        try:
            replacements = {
                '*': ' times ',
                '/': ' divided by ',
                '+': ' plus ',
                '-': ' minus ',
                '=': ' equals ',
                '^': ' to the power of ',
            }
            for symbol, word in replacements.items():
                text = text.replace(symbol, word)
            return text
        except Exception as e:
            self.logger.info(f"Error in preprocess_for_tts: {e}")
            return text

    def handle_question(self):
        try:
            self.stop_wake_word_listener()
            recognizer = sr.Recognizer()
            mic = sr.Microphone()
            with mic as source:
                recognizer.adjust_for_ambient_noise(source)
                try:
                    self.beep(880, 120)
                    self.logger.info("Ask your question...")
                    self.listening_state = "question"
                    audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
                except sr.WaitTimeoutError:
                    self.logger.info("No speech detected. Please try again.")
                    self.run_wake_word_listener()
                    return
            try:
                question = recognizer.recognize_google(audio)
                self.logger.info(f"Recognized question: {question}")
            except Exception as e:
                self.logger.info(f"Could not recognize question: {e}")
                self.run_wake_word_listener()
                return

            try:
                response = self.ai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    store=True,
                    messages=[
                        {"role": "system", "content": "When you write equations, use words for operators, e.g., 'times' instead of '*', 'divided by' instead of '/', etc."},
                        {"role": "user", "content": question}
                    ]
                )
                answer = response.choices[0].message.content.strip()
                self.logger.info(f"Answer: {answer}")
            except Exception as e:
                answer = "Sorry, I could not get an answer."
                self.logger.info(f"OpenAI error: {e}")

            if self.engine.isBusy():
                self.engine.stop()
            self.engine.say(answer, "Question Answer")
            self.engine.runAndWait()
            self.run_wake_word_listener()
        except Exception as e:
            self.logger.error(f"Error in handle_question: {e}")

    def draw_listening_indicator(self):
        try:
            t = pygame.time.get_ticks() / 1000.0
            pulse = 0.5 + 0.5 * math.sin(t * 2)
            min_radius = 2
            max_radius = 10
            radius = int(min_radius + (max_radius - min_radius) * pulse)
            if self.listening_state == "wake_word":
                color = (0, 255, 0)
            elif self.listening_state == "question":
                color = (255, 140, 0)
            else:
                return
            pygame.draw.circle(self.screen, color, (60, 60), radius)
        except Exception as e:
            self.logger.info(f"Error in draw_listening_indicator: {e}")

    def draw_debug_logs(self):
        try:
            if not self.debug_mode:
                return
            font = pygame.font.Font(None, 22)
            y = 50
            for line in list(self.log_buffer)[-15:]:
                surf = font.render(line, True, (255, 255, 0))
                self.screen.blit(surf, (10, y))
                y += 18
        except Exception as e:
            self.logger.info(f"Error in draw_debug_logs: {e}")

    def apply_settings(self):
        try:
            # Set voice and rate from settings
            if self.settings_voice_list:
                self.engine.setProperty("voice", self.settings_voice_list[self.settings_voice_index].id)
            if self.settings_voice_rate:
                self.engine.setProperty("rate", self.settings_voice_rate)
            self.logger.info(f"Applied TTS settings: voice={self.settings_voice_list[self.settings_voice_index].name}, rate={self.settings_voice_rate}")
        except Exception as e:
            self.logger.info(f"Error applying settings: {e}")

    def draw_settings_menu(self):
        try:
            menu_font = pygame.font.Font(None, 22)
            title_font = pygame.font.Font(None, 32)
            y = 20
            self.screen.fill((10, 10, 10))
            # Title
            title = title_font.render("Settings", True, (0, 255, 0))
            self.screen.blit(title, (self.SCREEN_WIDTH // 2 - title.get_width() // 2, y))
            y += 40

            # Voice selection (show current and total, allow scrolling)
            voice_label = menu_font.render("TTS Voice:", True, (255, 255, 255))
            self.screen.blit(voice_label, (30, y))
            if self.settings_voice_list:
                total_voices = len(self.settings_voice_list)
                idx = self.settings_voice_index
                voice = self.settings_voice_list[idx]
                voice_text = menu_font.render(f"{idx+1}/{total_voices}: {voice.name}", True, (0, 255, 0))
                self.screen.blit(voice_text, (150, y))
            y += 28

            # Rate slider (compact)
            rate_label = menu_font.render("Rate:", True, (255, 255, 255))
            self.screen.blit(rate_label, (30, y))
            rate_val = menu_font.render(str(self.settings_voice_rate), True, (0, 255, 0))
            self.screen.blit(rate_val, (90, y))
            slider_x = 140
            slider_y = y + 10
            slider_w = 120
            slider_h = 6
            pygame.draw.rect(self.screen, (80, 80, 80), (slider_x, slider_y, slider_w, slider_h))
            rate_range = self.settings_rate_max - self.settings_rate_min
            knob_pos = int(slider_x + ((self.settings_voice_rate - self.settings_rate_min) / rate_range) * slider_w)
            pygame.draw.circle(self.screen, (0, 255, 0), (knob_pos, slider_y + slider_h // 2), 8)
            y += 28

            # Recordings (show up to 7 at a time, scrollable)
            rec_label = menu_font.render("Recordings:", True, (255, 255, 255))
            self.screen.blit(rec_label, (30, y))
            y += 22
            rec_keys = list(self.recordings.keys())
            max_visible = 7
            rec_start = 0
            if self.settings_selected_recording and self.settings_selected_recording in rec_keys:
                idx = rec_keys.index(self.settings_selected_recording)
                if idx >= max_visible:
                    rec_start = idx - max_visible + 1
            for i, key in enumerate(rec_keys[rec_start:rec_start+max_visible]):
                color = (0, 255, 0) if self.settings_selected_recording == key else (180, 180, 180)
                msg = self.recordings[key][:20] + ("..." if len(self.recordings[key]) > 20 else "")
                rec_text = menu_font.render(f"{key}: {msg}", True, color)
                self.screen.blit(rec_text, (60, y + i * 20))
            y += max_visible * 20 + 4

            # Edit box for selected recording
            if self.settings_selected_recording:
                edit_label = menu_font.render("Edit:", True, (255, 255, 255))
                self.screen.blit(edit_label, (30, y))
                edit_box = pygame.Rect(80, y, 320, 22)
                pygame.draw.rect(self.screen, (40, 40, 40), edit_box)
                edit_text = menu_font.render(self.settings_edit_text[:40], True, (0, 255, 0))
                self.screen.blit(edit_text, (edit_box.x + 5, edit_box.y + 2))
                y += 26

            # Instructions (smaller font)
            instr_font = pygame.font.Font(None, 16)
            instr = [
                "UP/DOWN: Change voice   LEFT/RIGHT: Adjust rate",
                "TAB: Next recording   DEL: Delete   E: Edit   ENTER: Save   ESC: Close"
            ]
            for i, line in enumerate(instr):
                instr_text = instr_font.render(line, True, (180, 180, 180))
                self.screen.blit(instr_text, (30, self.SCREEN_HEIGHT - 40 + i * 16))
        except Exception as e:
            self.logger.info(f"Error drawing settings menu: {e}")

    def handle_settings_event(self, event):
        try:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.show_settings = False
                    self.apply_settings()
                elif event.key == pygame.K_UP:
                    if self.settings_voice_list:
                        self.settings_voice_index = (self.settings_voice_index - 1) % len(self.settings_voice_list)
                elif event.key == pygame.K_DOWN:
                    if self.settings_voice_list:
                        self.settings_voice_index = (self.settings_voice_index + 1) % len(self.settings_voice_list)
                elif event.key == pygame.K_LEFT:
                    self.settings_voice_rate = max(self.settings_rate_min, self.settings_voice_rate - self.settings_rate_step)
                elif event.key == pygame.K_RIGHT:
                    self.settings_voice_rate = min(self.settings_rate_max, self.settings_voice_rate + self.settings_rate_step)
                elif event.key == pygame.K_TAB:
                    rec_keys = list(self.recordings.keys())
                    if rec_keys:
                        if self.settings_selected_recording is None:
                            self.settings_selected_recording = rec_keys[0]
                        else:
                            idx = rec_keys.index(self.settings_selected_recording)
                            self.settings_selected_recording = rec_keys[(idx + 1) % len(rec_keys)]
                        self.settings_edit_text = self.recordings[self.settings_selected_recording]
                elif event.key == pygame.K_DELETE:
                    if self.settings_selected_recording:
                        del self.recordings[self.settings_selected_recording]
                        self.save_recordings()
                        self.settings_selected_recording = None
                        self.settings_edit_text = ""
                elif event.key == pygame.K_e:
                    if self.settings_selected_recording:
                        self.settings_edit_text = self.recordings[self.settings_selected_recording]
                elif event.key == pygame.K_RETURN:
                    if self.settings_selected_recording:
                        self.recordings[self.settings_selected_recording] = self.settings_edit_text
                        self.save_recordings()
                elif event.key == pygame.K_BACKSPACE:
                    if self.settings_selected_recording and self.settings_edit_text:
                        self.settings_edit_text = self.settings_edit_text[:-1]
                # Text input for editing
            elif event.type == pygame.TEXTINPUT:
                if self.settings_selected_recording:
                    self.settings_edit_text += event.text
        except Exception as e:
            self.logger.info(f"Error in handle_settings_event: {e}")

    def is_raspberry_pi(self):
        try:
            with open("/proc/device-tree/model") as f:
                return "Raspberry Pi" in f.read()
        except Exception:
            return False

    def get_pi_power_status(self):
        try:
            # Check for undervoltage
            throttled = subprocess.check_output(['vcgencmd', 'get_throttled']).decode()
            undervoltage = "0x1" in throttled or "0x50000" in throttled
            # Get voltage
            volts = subprocess.check_output(['vcgencmd', 'measure_volts']).decode().strip()
            return undervoltage, volts
        except Exception:
            return False, "Unavailable"

    def draw_pi_status(self):
        try:
            # Only show if in debug mode or currently under voltage
            if self.is_pi and (self.debug_mode or self.undervoltage_warning):
                font = pygame.font.Font(None, 24)
                y = self.SCREEN_HEIGHT - 60
                volt_text = font.render(f"Pi Voltage: {self.pi_voltage}", True, (255, 255, 0))
                self.screen.blit(volt_text, (10, y))
                if self.undervoltage_warning:
                    warn_text = font.render("!!! UNDERVOLTAGE DETECTED !!!", True, (255, 0, 0))
                    self.screen.blit(warn_text, (10, y + 30))
        except Exception as e:
            self.logger.info(f"Error drawing Pi status: {e}")

    def main(self):
        try:
            self.run_wake_word_listener()
            running = True
            mouth_anim_default = ")"
            mouth_anim = mouth_anim_default
            recording_thread = None
            last_transcribed_text = ""
            display_save_prompt = False
            self.engine.runAndWait()
            pi_status_counter = 0  # For periodic update
            while running:
                try:
                    for event in pygame.event.get():
                        # SETTINGS MENU TOGGLE (Ctrl+S)
                        if event.type == pygame.KEYDOWN and (event.key == pygame.K_s and (event.mod & pygame.KMOD_CTRL)):
                            self.show_settings = not self.show_settings
                            continue
                        if self.show_settings:
                            self.handle_settings_event(event)
                            continue
                        if event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_d and (event.mod & pygame.KMOD_CTRL):
                                self.debug_mode = not self.debug_mode
                                self.logger.info(f"Debug mode {'enabled' if self.debug_mode else 'disabled'}")
                            if display_save_prompt and self.awaiting_save:
                                if event.key == pygame.K_ESCAPE:
                                    self.logger.info("Save canceled by ESC")
                                    self.awaiting_save = False
                                    last_transcribed_text = ""
                                    display_save_prompt = False
                                elif event.key in [pygame.K_LSHIFT, pygame.K_RSHIFT]:
                                    self.stop_wake_word_listener()
                                    self.beep(880, 120)
                                    self.recording = True
                                    self.awaiting_save = False
                                    last_transcribed_text = ""
                                    display_save_prompt = False
                                    self.audio_data = []
                                    recording_thread = threading.Thread(target=self.record_audio)
                                    recording_thread.start()
                                    self.logger.info("Started recording")
                                    # Do NOT restart wake word here
                                elif event.key not in [pygame.K_LSHIFT, pygame.K_RSHIFT]:
                                    key_name = pygame.key.name(event.key)
                                    # Prevent saving to Ctrl key
                                    if key_name in ["left ctrl", "right ctrl", "ctrl"]:
                                        self.logger.info("Cannot save a recording to the Ctrl key.")
                                        self.awaiting_save = False
                                        last_transcribed_text = ""
                                        display_save_prompt = False
                                    elif last_transcribed_text and not last_transcribed_text.startswith("Error"):
                                        self.recordings[key_name] = last_transcribed_text
                                        self.logger.info(f"Saved to '{key_name}': {last_transcribed_text}")
                                        self.save_recordings()
                                        self.beep(440, 120)
                                    else:
                                        self.logger.info(f"Failed to save: {last_transcribed_text}")
                                    self.awaiting_save = False
                                    last_transcribed_text = ""
                                    display_save_prompt = False
                            else:
                                if event.key in [pygame.K_LSHIFT, pygame.K_RSHIFT] and not self.recording:
                                    self.stop_wake_word_listener()
                                    self.beep(880, 120)
                                    self.recording = True
                                    self.awaiting_save = False
                                    self.audio_data = []
                                    recording_thread = threading.Thread(target=self.record_audio)
                                    recording_thread.start()
                                    self.logger.info("Started recording")
                                    # Do NOT restart wake word here
                                elif event.key not in [pygame.K_LSHIFT, pygame.K_RSHIFT, pygame.K_ESCAPE, pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT]:
                                    key_name = pygame.key.name(event.key)
                                    if key_name in self.recordings:
                                        self.stop_wake_word_listener()
                                        self.logger.info(f"Playing '{key_name}': {self.recordings[key_name]}")
                                        self.play_message(self.recordings[key_name])
                                        self.run_wake_word_listener()
                                elif event.key == pygame.K_ESCAPE:
                                    self.logger.info("Exiting...")
                                    running = False
                                    self.stop_wake_word_listener()
                        elif event.type == pygame.KEYUP:
                            self.logger.info(f"KEYUP: key={event.key}, mod={event.mod}")
                            if event.key in [pygame.K_LSHIFT, pygame.K_RSHIFT] and self.recording:
                                self.recording = False
                                if recording_thread:
                                    recording_thread.join(timeout=2)  # Prevent lockup
                                self.beep(440, 120)
                                # Do NOT stop wake word here, already stopped
                                text = self.speech_to_text()
                                if text:
                                    self.logger.info(f"Recorded: {text}")
                                    last_transcribed_text = text
                                    self.awaiting_save = True
                                    display_save_prompt = True
                                else:
                                    self.logger.info(f"Recording failed: {text}")
                                    self.awaiting_save = False
                                    display_save_prompt = False
                                recording_thread = None
                                self.run_wake_word_listener()

                    # Periodically update Pi power status (every 1 second)
                    if self.is_pi:
                        pi_status_counter += 1
                        if pi_status_counter >= 20:  # 20 frames at 20 FPS = 1 second
                            self.undervoltage_warning, self.pi_voltage = self.get_pi_power_status()
                            pi_status_counter = 0

                    if self.show_settings:
                        self.draw_settings_menu()
                    else:
                        if self.is_talking:
                            mouth_anim = self.FACE_NORMAL if mouth_anim in [self.FACE_IDLE, self.FACE_WAIT] else self.FACE_WAIT
                        else:
                            mouth_anim = self.FACE_IDLE

                        self.draw_background()
                        self.draw_face(mouth=mouth_anim)
                        self.draw_interference()
                        self.draw_listening_indicator()
                        self.draw_debug_logs()
                        self.draw_pi_status()  # <-- Add this line

                        if display_save_prompt and last_transcribed_text:
                            prompt_font = pygame.font.Font(None, 36)
                            prompt_text = f"\"{last_transcribed_text}\""
                            instr_text = "Press a button to save, or ESC to cancel."
                            prompt_surf = prompt_font.render(prompt_text, True, (0, 255, 0))
                            instr_surf = prompt_font.render(instr_text, True, (0, 255, 0))
                            prompt_rect = prompt_surf.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2 + 80))
                            instr_rect = instr_surf.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2 + 120))
                            bg_color = (0, 40, 0)
                            padding = 16
                            pygame.draw.rect(
                                self.screen, bg_color,
                                (prompt_rect.left - padding, prompt_rect.top - padding,
                                 prompt_rect.width + 2 * padding, prompt_rect.height + 2 * padding)
                            )
                            pygame.draw.rect(
                                self.screen, bg_color,
                                (instr_rect.left - padding, instr_rect.top - padding,
                                 instr_rect.width + 2 * padding, instr_rect.height + 2 * padding)
                            )
                            self.screen.blit(prompt_surf, prompt_rect)
                            self.screen.blit(instr_surf, instr_rect)

                    pygame.display.flip()
                    self.clock.tick(20)

                except pygame.error as e:
                    self.logger.error(f"Pygame error: {e}")
                except Exception as e:
                    self.logger.error(f"Unexpected error: {e}")

            self.save_recordings()
            pygame.quit()
            sys.exit()
        except Exception as e:
            print(f"Fatal error in main: {e}")

if __name__ == "__main__":
    OldTV().main()
