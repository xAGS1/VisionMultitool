# 🧠 VisionMultitool

**VisionMultitool** is a multi-functional computer vision GUI tool built with Python. It integrates OCR text extraction from highlighted regions, object detection with YOLOv8, and a custom image classification model – all accessible via a user-friendly Tkinter interface.

---

## 🚀 Features

### 1. Highlight Text Extractor
- Use **EasyOCR** to detect text in images.
- Extract only the **highlighted portions** using color selection.
- Adjustable HSV **sensitivity range** for fine-tuning.
- Optional **spell checking** using `pyspellchecker`.
- **Text-to-speech** support with `pyttsx3` *(optional)*.
- Copy extracted text directly to clipboard.

### 2. Object Detection (YOLOv8)
- Upload an image and detect objects using a **YOLOv8** model.
- Start **live detection** using an IP camera (like your phone).
- Fully integrated with **Ultralytics YOLOv8**.
- ✅ **Trained on a custom dataset** with the following classes:
  - `cutter`, `dallah`, `nail clippers`, `sprinkler`, `terminal block`
- 🧪 **Data augmentation** was applied during training to improve robustness.

### 3. Simple Image Classifier
- Uses a custom lightweight **CNN model** for image classification.
- Loads `.pth` weights with structured YAML configs.
- Automatically applies the same preprocessing used during training.
- ✅ **Trained on a custom dataset with 2 classes**:
  - `stapler`
  - `hishambook`
- 🧪 **Data augmentation** techniques were applied to increase generalization.


---

## 🧰 Requirements

- Python 3.8+
- Works on Windows, macOS, and Linux (with minor tweaks)
- Optional: GPU support for YOLOv8 via CUDA

---

## 📦 Installation

### 1. Clone the repository

    git clone https://github.com/xAGS1/VisionMultitool.git
    cd VisionMultitool


### 2. Install dependencies

    pip install -r requirements.txt


### 3. Install YOLOv8 (if not already)

    pip install ultralytics


### 4. Optional: Install Text-to-Speech

    pip install pyttsx3


---

## 📁 Directory Structure

- `VisionMultitool/`
  - `main.py` — Entry point (contains all GUI logic)
  - `yolov8n-project/`
    - `best.pt` — YOLOv8 model weights
  - `ourMODEL/`
    - `best_model.pth` — CNN classifier weights
    - `model.yaml` — CNN architecture config
    - `hyp.yaml` — Hyperparameters (e.g. image size)
    - `data.yaml` — Dataset directory structure


---

## 🖼️ Highlight Color Picking

To extract **only highlighted text**:

1. Click **📷 Upload** and select your image.
2. Click **🎨 Pick Color** and select the highlighted area.
3. Adjust sensitivity if needed.
4. Click **🔍 Extract** to get text from highlighted regions.

You can optionally enable spell checking or read the text aloud.

---

## 🌐 Live Detection Setup (Optional)

To use **live object detection** via your phone:

1. Install an IP camera app such as [IP Webcam (Android)](https://play.google.com/store/apps/details?id=com.pas.webcam).
2. Start the camera and note the IP address and port (e.g. \`192.168.0.101:8080\`).
3. Enter the IP in the GUI and click **📡 Start Live Detection**.

Press **Q** to exit the live detection window.

---

## 🛠️ Built With

- [OpenCV](https://opencv.org/)
- [EasyOCR](https://github.com/JaidedAI/EasyOCR)
- [Ultralytics YOLOv8](https://docs.ultralytics.com/)
- [PyTorch](https://pytorch.org/)
- [Tkinter](https://docs.python.org/3/library/tkinter.html)
- [Pillow (PIL)](https://python-pillow.org/)
- [pyttsx3](https://pypi.org/project/pyttsx3/)
- [pyspellchecker](https://pypi.org/project/pyspellchecker/)

---

## 📌 Notes

- The application is **offline-first** – no internet is required after installing dependencies.
- **Ensure file paths** to models are correctly set in \`main.py\`.
- YOLOv8 and PyTorch will **automatically use GPU** if available.


---

## 🤝 Contributing

Pull requests are welcome! Feel free to fork this repo and submit PRs for:

- New features
- Bug fixes
- UI improvements
- Performance optimizations

---

## 📜 License

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.

---

## 💬 Acknowledgements

- [JaidedAI/EasyOCR](https://github.com/JaidedAI/EasyOCR)
- [Ultralytics](https://github.com/ultralytics/ultralytics) 
