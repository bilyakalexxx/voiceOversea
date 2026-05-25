# 👁️ voiceOversea

An online, assistive smartphone application built in Python that leverages high-performance remote Vision-Language Models to provide spoken scene descriptions for blind and visually impaired individuals.

## 🚀 Features
- **Multilingual Native Descriptions:** Delivers instantaneous environmental layout feedback in the user's preferred spoken language.
- **Gesture-Friendly Interface:** Built with a high-contrast, full-screen gesture UI engineered specifically for blind user interactions.

## 🧠 How It Works
1. **Capture:** The user taps the full-screen interface to fire the native camera hardware.
2. **Transfer:** The local picture file is packaged and sent over an online network connection.
3. **Analyze:** A remote server running an advanced Vision-Language Model (`Qwen2.5-VL`) processes the pixels in less than a second.
4. **Vocalize:** The server sends a concise text description back, which is spoken out loud immediately via the device's native Text-to-Speech (TTS) engine.

## 🛠️ Tech Stack
- **Frontend App:** Python, Kivy, Plyer
- **Backend AI Engine:** Ollama (`qwen2.5vl:3b` running on NVIDIA RTX 4090 Hardware)
- **Audio Output:** Device Native Accessibility Voice Synthesis

## 🎯 Target Goal
To establish a fast, online, high-performance, visual guidance network.
Create an assistive AI tool that provides not just information, but meaningful and human-like environmental descriptions.