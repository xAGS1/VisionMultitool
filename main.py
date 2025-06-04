import cv2
import easyocr
import numpy as np
import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox, ttk
from spellchecker import SpellChecker
import pyperclip
from PIL import Image, ImageTk
import time
import platform
import threading
from queue import Queue, Empty
import traceback
import torch
import torch.nn as nn
from pathlib import Path
# ""

path = Path(__file__).resolve().parent / 'yolov8n-project' / 'best.pt'

try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False
    print("Warning: pyttsx3 not found. 'Read Aloud' disabled. Install with: pip install pyttsx3")




# Core Logic Class
class HighlightDetector:
    def __init__(self):
        print("Initializing EasyOCR Reader...")
        self.reader = easyocr.Reader(['en'])
        print("EasyOCR Ready.")
        self.highlight_color = None
        self.color_range = 20
        self.spell = SpellChecker()

    def set_highlight_color(self, hsv_color):
        self.highlight_color = hsv_color

    def set_color_range(self, range_val):
        self.color_range = range_val

    def process_image(self, image_path, debug=False, enable_spell_check=True):
        if self.highlight_color is None:
            return "⚠️ Error: No highlight color selected.", 0
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Unable to load image: {image_path}")

            height, width = image.shape[:2]
            max_dim = 2500
            if max(height, width) > max_dim:
                scale = max_dim / max(height, width)
                image = cv2.resize(image, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)

            hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            h, s, v = self.highlight_color
            h_range = self.color_range
            s_min = max(30, int(s * 0.5), int(s - 60))
            v_min = max(50, int(v * 0.5), int(v - 80))
            lower_color = np.array([max(int(h - h_range), 0), s_min, v_min], dtype=np.uint8)
            upper_color = np.array([min(int(h + h_range), 179), 255, 255], dtype=np.uint8)

            if debug:
                print(f"Using HSV range: Lower={lower_color}, Upper={upper_color}")

            result = self.reader.readtext(image, paragraph=False, batch_size=8)
            if debug:
                print(f"OCR detected {len(result)} text blocks.")

            highlighted_text_fragments = []
            accepted_confidences = []
            debug_image_copy = image.copy() if debug else None

            for (bbox, text, conf) in result:
                pts = np.array(bbox, dtype=np.int32)
                x_min, y_min = np.min(pts[:, 0]), np.min(pts[:, 1])
                x_max, y_max = np.max(pts[:, 0]), np.max(pts[:, 1])
                x_min, y_min = max(0, x_min), max(0, y_min)
                x_max, y_max = min(image.shape[1], x_max), min(image.shape[0], y_max)

                if x_min >= x_max or y_min >= y_max:
                    continue

                sub_hsv = hsv_image[y_min:y_max, x_min:x_max]
                if sub_hsv.size == 0:
                    continue

                mask = cv2.inRange(sub_hsv, lower_color, upper_color)
                highlight_ratio = cv2.countNonZero(mask) / mask.size if mask.size > 0 else 0

                min_highlight_ratio = 0.25
                min_confidence = 0.4

                if highlight_ratio >= min_highlight_ratio and conf >= min_confidence:
                    highlighted_text_fragments.append(text)
                    accepted_confidences.append(conf)
                    if debug:
                        cv2.polylines(debug_image_copy, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
                elif debug:
                    cv2.polylines(debug_image_copy, [pts], isClosed=True, color=(0, 0, 255), thickness=1)

            if debug and debug_image_copy is not None:
                h_debug, w_debug = debug_image_copy.shape[:2]
                if max(h_debug, w_debug) > 1000:
                    scale_debug = 1000 / max(h_debug, w_debug)
                    debug_image_display = cv2.resize(debug_image_copy, (int(w_debug * scale_debug), int(h_debug * scale_debug)))
                else:
                    debug_image_display = debug_image_copy
                cv2.imshow("Debug View - Highlighted Text Regions", debug_image_display)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            if not highlighted_text_fragments:
                return "⚠️ No highlighted text found.", 0

            full_text = ' '.join(highlighted_text_fragments)

            if enable_spell_check:
                words = full_text.split()
                corrected_words = [self.spell.correction(w) or w for w in words if w]
                final_text = ' '.join(corrected_words)
            else:
                final_text = full_text

            avg_confidence = np.mean(accepted_confidences) if accepted_confidences else 0

            return final_text.strip(), avg_confidence

        except Exception as e:
            print(f"Error during processing: {e}\n{traceback.format_exc()}")
            return f"⚠️ Error: {str(e)}", 0


#  GUI
class HighlightApp:
    def __init__(self, root, go_back_callback):
        self.root = root
        self.go_back_callback = go_back_callback
        self.image_path = None
        self.original_image_for_picker = None
        self.detector = HighlightDetector()
        self.result_queue = Queue()
        self.root = root
        self.root.title("Highlight Text Extractor")
        self.image_path = None
        self.original_image_for_picker = None
        self.detector = HighlightDetector()
        self.result_queue = Queue()

        self.root.geometry("850x700")
        self.root.configure(bg="#f0f0f0")
        self.root.minsize(650, 550)

        main_frame = tk.Frame(root, bg="#f0f0f0")
        main_frame.pack(pady=10, padx=20, fill=tk.BOTH, expand=True)

        self.preview_label = tk.Label(main_frame, bg="white", bd=2, relief="groove", text="Upload image for preview")
        self.preview_label.pack(pady=(0, 10), fill=tk.BOTH, expand=True)

        control_btn_frame_outer = tk.Frame(main_frame, bg="#f0f0f0")
        control_btn_frame_outer.pack(pady=5)
        control_btn_frame_inner = tk.Frame(control_btn_frame_outer, bg="#f0f0f0")
        control_btn_frame_inner.pack()

        action_btn_frame_outer = tk.Frame(main_frame, bg="#f0f0f0")
        action_btn_frame_outer.pack(pady=5)
        action_btn_frame_inner = tk.Frame(action_btn_frame_outer, bg="#f0f0f0")
        action_btn_frame_inner.pack()

        self.upload_btn = tk.Button(control_btn_frame_inner, text="📷 Upload", command=self.upload_image, font=("Arial", 10), bg="#4CAF50", fg="white", width=15)
        self.upload_btn.pack(side=tk.LEFT, padx=5, pady=5)
        self.pick_color_btn = tk.Button(control_btn_frame_inner, text="🎨 Pick Color", command=self.pick_color_from_image, font=("Arial", 10), bg="#2196F3", fg="white", width=15)
        self.pick_color_btn.pack(side=tk.LEFT, padx=5, pady=5)
        self.extract_btn = tk.Button(control_btn_frame_inner, text="🔍 Extract", command=self.start_extraction_thread, font=("Arial", 10, "bold"), bg="#FF9800", fg="white", width=15)
        self.extract_btn.pack(side=tk.LEFT, padx=5, pady=5)

        settings_frame = tk.Frame(main_frame, bg="#f0f0f0")
        settings_frame.pack(fill=tk.X, pady=5)
        self.debug_var = tk.BooleanVar()
        tk.Checkbutton(settings_frame, text="Debug", variable=self.debug_var, bg="#f0f0f0").pack(side=tk.LEFT, padx=(10, 5))

        self.spell_check_var = tk.BooleanVar(value=False)  # Default: not checked
        self.spell_checkbox = tk.Checkbutton(settings_frame, text="Enable Spell Checking",
                                     variable=self.spell_check_var,
                                     onvalue=0, offvalue=1,
                                     bg="#f0f0f0")
        self.spell_checkbox.pack(side=tk.LEFT, padx=10)


        tk.Label(settings_frame, text="Sensitivity:", bg="#f0f0f0").pack(side=tk.LEFT, padx=5)
        self.color_range_slider = tk.Scale(settings_frame, from_=5, to=50, orient=tk.HORIZONTAL, command=lambda v: self.detector.set_color_range(int(v)), bg="#f0f0f0", highlightthickness=0, length=150)
        self.color_range_slider.set(self.detector.color_range)
        self.color_range_slider.pack(side=tk.LEFT, padx=5)

        output_frame = tk.Frame(main_frame, bg="#f0f0f0")
        output_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        self.output_box = scrolledtext.ScrolledText(output_frame, height=8, wrap=tk.WORD, font=("Arial", 11), bd=2, relief="groove")
        self.output_box.pack(fill=tk.BOTH, expand=True)

        self.precision_var = tk.StringVar()
        self.precision_var.set("Prediction Precision: N/A")
        self.precision_label = tk.Label(output_frame, textvariable=self.precision_var, font=("Arial", 10), bg="#f0f0f0", anchor="w")
        self.precision_label.pack(fill=tk.X, pady=(2, 0))

        self.copy_btn = tk.Button(action_btn_frame_inner, text="📋 Copy", command=self.copy_to_clipboard, font=("Arial", 10), bg="#607D8B", fg="white", width=15)
        self.copy_btn.pack(side=tk.LEFT, padx=5, pady=5)
        
        read_state = tk.NORMAL if PYTTSX3_AVAILABLE else tk.DISABLED
        read_bg = "#FF5722" if PYTTSX3_AVAILABLE else "#cccccc"
        self.read_btn = tk.Button(action_btn_frame_inner, text="🔊 Read", command=self.read_aloud, font=("Arial", 10), bg=read_bg, fg="white", width=15, state=read_state)
        self.read_btn.pack(side=tk.LEFT, padx=5, pady=5)

        self.progress_bar = ttk.Progressbar(main_frame, orient=tk.HORIZONTAL, length=300, mode='indeterminate')
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        tk.Label(root, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W, bg="#e0e0e0").pack(side=tk.BOTTOM, fill=tk.X)
        self.back_btn = tk.Button(main_frame, text="🔙 Back", command=self.go_back, bg="#9E9E9E", fg="white", width=15)
        self.back_btn.pack(pady=10)

    #  the Helper Methods  
    def go_back(self):
        for widget in self.root.winfo_children():
            widget.destroy()
        self.go_back_callback()


    def copy_to_clipboard(self):
        text = self.output_box.get("1.0", tk.END).strip()
        if text and not text.startswith("⚠️"):
            pyperclip.copy(text)
            self.update_status("Copied to clipboard.")
        else:
            messagebox.showwarning("Nothing to Copy", "There is no valid extracted text to copy.", parent=self.root)

    def update_status(self, message):
        self.root.after(0, self.status_var.set, message)

    def set_buttons_state(self, state):
        for btn in [self.upload_btn, self.pick_color_btn, self.extract_btn, self.copy_btn, self.read_btn]:
            if btn == self.read_btn and not PYTTSX3_AVAILABLE:
                continue
            btn.config(state=state)


    def upload_image(self):
        path = filedialog.askopenfilename(title="Select Image", filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tiff"), ("All", "*.*")])
        if not path:
            return
        self.image_path = path
        try:
            self.original_image_for_picker = cv2.imread(self.image_path)
            if self.original_image_for_picker is None:
                raise ValueError("cv2.imread failed")
            img_pil = Image.open(path)
            max_w = self.preview_label.winfo_width() if self.preview_label.winfo_width() > 50 else 600
            max_h = self.preview_label.winfo_height() if self.preview_label.winfo_height() > 50 else 400
            img_pil.thumbnail((max_w - 10, max_h - 10), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img_pil)
            self.preview_label.config(image=photo, text="")
            self.preview_label.image = photo
            self.output_box.delete(1.0, tk.END)
            self.detector.highlight_color = None
            self.update_status(f"Loaded: {path.split('/')[-1]}. Pick color.")
        except Exception as e:
            messagebox.showerror("Error Loading Image", f"Failed to load image: {path}\n{str(e)}", parent=self.root)
            self.image_path = None
            self.original_image_for_picker = None
            self.preview_label.config(image=None, text="Error loading image.")
            self.preview_label.image = None
            self.update_status("Error loading image.")

    def pick_color_from_image(self):
        if self.original_image_for_picker is None:
            return messagebox.showwarning("No Image", "Upload image first.", parent=self.root)
        self.update_status("Click on highlight color in the new window...")

        image_display = self.original_image_for_picker.copy()
        height, width = image_display.shape[:2]
        max_dim = 1000
        if max(height, width) > max_dim:
            scale = max_dim / max(height, width)
            image_display = cv2.resize(image_display, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)

        picker_window_name = "Pick Highlight Color (Click Color, ESC to Cancel)"
        cv2.namedWindow(picker_window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(picker_window_name, min(width, 900), min(height, 700))
        selected_hsv = None

        def mouse_callback(event, x, y, flags, param):
            nonlocal selected_hsv
            if event == cv2.EVENT_LBUTTONDOWN:
                try:
                    if 0 <= y < image_display.shape[0] and 0 <= x < image_display.shape[1]:
                        bgr = image_display[y, x]
                        selected_hsv = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)[0][0]
                        cv2.destroyWindow(picker_window_name)
                except Exception as e:
                    print(f"Mouse callback error: {e}")

        cv2.imshow(picker_window_name, image_display)
        cv2.setMouseCallback(picker_window_name, mouse_callback)
        key = cv2.waitKey(0)
        if key == 27:
            cv2.destroyWindow(picker_window_name)
        cv2.destroyAllWindows()

        if selected_hsv is not None:
            self.detector.set_highlight_color(selected_hsv)
            self.update_status(f"Color set (HSV: {selected_hsv}). Ready to Extract.")
            messagebox.showinfo("Color Selected", f"Highlight color set (HSV: {selected_hsv}).\nAdjust sensitivity if needed.", parent=self.root)
        else:
            self.update_status("Color picking cancelled or failed.")

    def start_extraction_thread(self):
        if not self.image_path:
            return messagebox.showwarning("No Image", "Upload image first.", parent=self.root)
        if self.detector.highlight_color is None:
            return messagebox.showwarning("No Color", "Pick highlight color first.", parent=self.root)

        self.update_status("Processing...")
        self.set_buttons_state(tk.DISABLED)
        self.output_box.delete(1.0, tk.END)
        self.progress_bar.pack(pady=(5, 0), fill=tk.X, padx=10)
        self.progress_bar.start(10)
        threading.Thread(target=self._run_extraction, daemon=True).start()
        self.root.after(100, self.check_extraction_queue)

    def _run_extraction(self):
        start_time = time.time()
        try:
            enable_spell_check = (self.spell_check_var.get() == 0)

            text_result = self.detector.process_image(self.image_path, self.debug_var.get(), enable_spell_check)
            self.result_queue.put(text_result)
        except Exception as e:
            error_message = f"⚠️ Thread Error: {str(e)}"
            print(f"Extraction thread error: {e}\n{traceback.format_exc()}")
            self.result_queue.put(error_message)
        finally:
            print(f"Extraction thread finished in {time.time() - start_time:.2f}s.")

    def check_extraction_queue(self):
        try:
            result, avg_conf = self.result_queue.get_nowait()
            self.progress_bar.stop()
            self.progress_bar.pack_forget()
            self.output_box.insert(tk.END, result)

            if avg_conf > 0:
                self.precision_var.set(f"Prediction Precision: {avg_conf*100:.1f}%")
            else:
                self.precision_var.set("Prediction Precision: N/A")

            char_count = len(result) if not result.startswith("⚠️") else 0
            self.update_status(f"Extraction complete | {char_count} chars found.")
            self.set_buttons_state(tk.NORMAL)

            if result.startswith("⚠️"):
                messagebox.showerror("Extraction Error", result, parent=self.root)

        except Empty:
            self.root.after(100, self.check_extraction_queue)
        except Exception as e:
            print(f"Queue check error (real error): {e}")
            self.progress_bar.stop()
            self.progress_bar.pack_forget()
            self.set_buttons_state(tk.NORMAL)
            self.update_status("Error checking results.")

    def read_aloud(self):
        if not PYTTSX3_AVAILABLE:
            return messagebox.showerror("Unavailable", "Install 'pyttsx3' for Read Aloud.", parent=self.root)

        text = self.output_box.get("1.0", tk.END).strip()
        if text and not text.startswith("⚠️"):
            self.update_status("Reading aloud...")

            def speak():
                try:
                    tts_engine = pyttsx3.init()
                    tts_engine.say(text)
                    tts_engine.runAndWait()
                    tts_engine.stop()
                    self.update_status("Finished reading.")
                except Exception as e:
                    messagebox.showerror("TTS Error", f"Could not read aloud:\n{str(e)}", parent=self.root)
                    self.update_status("TTS engine error.")
                finally:
                    try:
                        if self.read_btn and self.read_btn.winfo_exists():
                             self.read_btn.config(state=tk.NORMAL)
                    except:
                           pass

            self.read_btn.config(state=tk.DISABLED)
            threading.Thread(target=speak, daemon=True).start()
        elif text.startswith("⚠️"):
            messagebox.showwarning("Cannot Read", "Cannot read error messages.", parent=self.root)
        else:
            messagebox.showwarning("Empty", "No text to read.", parent=self.root)

# -model  Class
class DetectApp:
    def __init__(self, root, go_back_callback):
        self.root = root
        self.go_back_callback = go_back_callback
        self.image_path = None
        self.model = None

        self.frame = tk.Frame(root, bg="#f0f0f0")
        self.frame.pack(fill=tk.BOTH, expand=True)

        self.top_buttons = tk.Frame(self.frame, bg="#f0f0f0")
        self.top_buttons.pack(pady=10)

        self.upload_btn = tk.Button(self.top_buttons, text="📷 Upload Image", command=self.upload_image, bg="#4CAF50", fg="white", width=20)
        self.upload_btn.pack(side=tk.LEFT, padx=5)

        self.predict_btn = tk.Button(self.top_buttons, text="🔍 Predict Image", command=self.predict, bg="#FF9800", fg="white", width=20)
        self.predict_btn.pack(side=tk.LEFT, padx=5)

        self.back_btn = tk.Button(self.top_buttons, text="🔙 Back", command=self.go_back, bg="#9E9E9E", fg="white", width=20)
        self.back_btn.pack(side=tk.LEFT, padx=5)

     
        self.ip_frame = tk.Frame(self.frame, bg="#f0f0f0")
        self.ip_frame.pack(pady=5)

        tk.Label(self.ip_frame, text="Phone IP Address:", bg="#f0f0f0").pack(side=tk.LEFT, padx=(5, 2))
        self.ip_entry = tk.Entry(self.ip_frame)
        self.ip_entry.insert(0, "192.168.1.1")  # default  IP
        self.ip_entry.pack(side=tk.LEFT, padx=5)

        self.live_btn = tk.Button(self.ip_frame, text="📡 Start Live Detection", command=self.start_live_detection, bg="#2196F3", fg="white")
        self.live_btn.pack(side=tk.LEFT, padx=5)

        self.preview_label = tk.Label(self.frame, bg="white", bd=2, relief="groove", text="Upload image for preview or start live detection")
        self.preview_label.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        # Load theeeee YOLOv8 model
        self.load_model()

    def load_model(self):
        try:
            from ultralytics import YOLO
            self.model = YOLO(path)  # Your path here of the model
        except Exception as e:
            messagebox.showerror("Model Load Error", f"Failed to load YOLOv8 model.\n{str(e)}", parent=self.root)
    def upload_image(self):
        path = filedialog.askopenfilename(title="Select Image", filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tiff"), ("All", "*.*")])
        if not path:
            return
        self.image_path = path
        img_pil = Image.open(path)
        img_pil.thumbnail((600, 400))
        photo = ImageTk.PhotoImage(img_pil)
        self.preview_label.config(image=photo, text="")
        self.preview_label.image = photo

    def predict(self):  
        if not self.image_path:
            messagebox.showwarning("No Image", "Please upload an image first.", parent=self.root)
            return
        try:
            results = self.model.predict(
                source=self.image_path,
                save=False,
                imgsz=640,
                conf=0.25
            )

            if len(results[0].boxes) == 0:
                messagebox.showinfo("No Detections", "No objects detected in the image.", parent=self.root)
                return

            pred_img = results[0].plot()

            img_pil = Image.fromarray(pred_img)
            img_pil.thumbnail((600, 400))
            photo = ImageTk.PhotoImage(img_pil)

            self.preview_label.config(image=photo, text="", compound="center")
            self.preview_label.image = photo
        except Exception as e:
            messagebox.showerror("Prediction Error", f"Failed to predict.\n{str(e)}", parent=self.root)


    def start_live_detection(self):
        ip = self.ip_entry.get().strip()  # Get the IP address from the textbox
        if not ip:
            messagebox.showwarning("Missing IP", "Please enter your phone IP address!", parent=self.root)
            return
        threading.Thread(target=self.live_detection_loop, args=(ip,), daemon=True).start()

    def live_detection_loop(self, ip):
        try:
            PORT = '8080'  #  make this dynamic  later maybe??
            stream_url = f"http://{ip}:{PORT}/video"

            cap = cv2.VideoCapture(stream_url)
            if not cap.isOpened():
                raise Exception(f"Could not open stream at {stream_url}")

            cv2.namedWindow("Live YOLOv8 Detection", cv2.WINDOW_NORMAL)

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.resize(frame, (640, 480))

                results = self.model.predict(source=frame, imgsz=640, conf=0.5, save=False)
                annotated_frame = results[0].plot()

                cv2.imshow("Live YOLOv8 Detection", annotated_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()

        except Exception as e:
            print(f"Live detection error: {e}")
            messagebox.showerror("Live Detection Error", str(e), parent=self.root)

    def go_back(self):
        self.frame.destroy()
        self.go_back_callback()
        
class SimpleClassifierApp:
    def __init__(self, root, go_back_callback):
        self.root = root
        self.go_back_callback = go_back_callback
        self.image_path = None

        # ─── ensure transform always exists ───
        self.transform = None

        # ─── UI setup ───
        self.frame = tk.Frame(root, bg="#f0f0f0")
        self.frame.pack(fill=tk.BOTH, expand=True)

        # Top Buttons
        top_buttons = tk.Frame(self.frame, bg="#f0f0f0")
        top_buttons.pack(pady=10)
        self.upload_btn = tk.Button(top_buttons, text="📷 Upload Image",
                                    command=self.upload_image,
                                    bg="#4CAF50", fg="white", width=20)
        self.upload_btn.pack(side=tk.LEFT, padx=5)
        self.predict_btn = tk.Button(top_buttons, text="🔍 Predict",
                                     command=self.predict,
                                     bg="#FF9800", fg="white", width=20)
        self.predict_btn.pack(side=tk.LEFT, padx=5)
        self.back_btn = tk.Button(top_buttons, text="🔙 Back",
                                  command=self.go_back,
                                  bg="#9E9E9E", fg="white", width=20)
        self.back_btn.pack(side=tk.LEFT, padx=5)

        # Image preview
        self.preview_label = tk.Label(self.frame, bg="white", bd=2, relief="groove",
                                      text="Upload an image for classification")
        self.preview_label.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        # Prediction text
        self.result_label = tk.Label(self.frame, text="Prediction: N/A",
                                     font=("Arial", 16), bg="#f0f0f0")
        self.result_label.pack(pady=10)

        # Now load model + transforms
        self.load_model()


    def load_model(self):
        try:
            import yaml
            from pathlib import Path
            from torchvision import transforms

            base_dir = Path(__file__).resolve().parent

           
            self.classes = ["hishambook", "Stapler"]

            # Load configuration
            WEIGHTS = base_dir / 'CnnMODEL' / 'best_model.pth'
            cfg     = yaml.safe_load(open(base_dir / 'CnnMODEL' / 'model.yaml'))
            hyp     = yaml.safe_load(open(base_dir / 'CnnMODEL' / 'hyp.yaml'))

           
            cfg['nc'] = len(self.classes)

          
            class TinyClassifier(nn.Module):
                def __init__(self, cfg):
                    super().__init__()
                    layers = []
                    in_ch = 3  # RGB input
                    for block in cfg['backbone']:
                        _, out_ch, k, s = block
                        layers += [
                            nn.Conv2d(in_ch, out_ch, k, s, k // 2),
                            nn.BatchNorm2d(out_ch),
                            nn.ReLU()
                        ]
                        in_ch = out_ch
                    self.features = nn.Sequential(*layers)
                    self.pool = nn.AdaptiveAvgPool2d((1, 1))
                    self.classifier = nn.Linear(in_ch, cfg['nc'])

                def forward(self, x):
                    x = self.pool(self.features(x))
                    return self.classifier(x.view(x.size(0), -1))

            # Load the model
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model = TinyClassifier(cfg).to(self.device)
            self.model.load_state_dict(torch.load(WEIGHTS, map_location=self.device))
            self.model.eval()

            # Preprocessing
            self.transform = transforms.Compose([
                transforms.Resize((int(hyp['imgsz']), int(hyp['imgsz']))),
                transforms.ToTensor(),
            ])

        except Exception as e:
            messagebox.showerror("Model Load Error", f"Failed to load classifier model:\n{e}", parent=self.root)






    def upload_image(self):
        path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Images","*.png *.jpg *.jpeg *.bmp *.tiff"),("All","*.*")])
        if not path:
            return
        self.image_path = path
        img_pil = Image.open(path)
        img_pil.thumbnail((600,400))
        photo = ImageTk.PhotoImage(img_pil)
        self.preview_label.config(image=photo, text="")
        self.preview_label.image = photo
        self.result_label.config(text="Prediction: N/A")

    def predict(self):
        if not self.image_path:
            return messagebox.showwarning("No Image",
                "Please upload an image first.", parent=self.root)

        if self.transform is None:
            return messagebox.showerror("Model Not Ready",
                "Classifier not loaded properly—please check logs.", parent=self.root)

        try:
            # preprocess
            img = Image.open(self.image_path).convert('RGB')
            tensor = self.transform(img).unsqueeze(0).to(self.device)

            # inference
            with torch.no_grad():
                out = self.model(tensor)
                idx = int(out.argmax(dim=1))
                cls = self.classes[idx]

            # display
            self.result_label.config(text=f"Prediction: {cls}")

        except Exception as e:
            messagebox.showerror("Prediction Error",
                f"Failed to predict:\n{e}", parent=self.root)

    def go_back(self):
        self.frame.destroy()
        self.go_back_callback()




# --- Main Menu ---
class MainApp:
    
    def __init__(self, root):
        self.root = root
        self.root.title("VisionMultitool")
        self.root.geometry("900x700")
        self.root.configure(bg="#e0e0e0")

        self.menu_frame = tk.Frame(root, bg="#e0e0e0")
        self.menu_frame.pack(expand=True)

        tk.Label(self.menu_frame, text="Choose a Tool", font=("Arial", 24), bg="#e0e0e0").pack(pady=30)

        tk.Button(self.menu_frame, text="Highlight Text Extractor ", command=self.launch_highlight_extractor, width=30, height=2, bg="#4CAF50", fg="white", font=("Arial", 14)).pack(pady=10)
        tk.Button(self.menu_frame, text="Object Detection (YOLOv8)", command=self.launch_yolo, width=30, height=2, bg="#2196F3", fg="white", font=("Arial", 14)).pack(pady=10)
        tk.Button(self.menu_frame, text="Simple Image Classifier", command=self.launch_simple_classifier, width=30, height=2, bg="#9C27B0", fg="white", font=("Arial", 14)).pack(pady=10)
    def launch_highlight_extractor(self):
        self.menu_frame.destroy()
        HighlightApp(self.root, self.back_to_menu)

    def launch_yolo(self):
        self.menu_frame.destroy()
        DetectApp(self.root, self.back_to_menu)

    def back_to_menu(self):
        for widget in self.root.winfo_children():
            widget.destroy()
        self.__init__(self.root)

    def launch_simple_classifier(self):
        self.menu_frame.destroy()
        SimpleClassifierApp(self.root, self.back_to_menu)
    


if __name__ == "__main__":
    root = tk.Tk()
    app = MainApp(root)
    root.mainloop()