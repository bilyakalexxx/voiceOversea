[app]

# (str) Title of your application
title = voiceOversea

# (str) Package name
package.name = voiceoversea

# (str) Package domain (needed for android/ios packaging)
package.domain = org.assistive

# (str) Source code directory where main.py resides (current directory)
source.dir = .

# (str) Application version
version = 0.1

# (list) Source files to include (let's include py and assets)
source.include_exts = py,png,jpg,kv,atlas,json

# (list) Application requirements
# (We removed the heavy 'ollama' package. 'requests' is completely pure and safe)
requirements = python3, kivy==2.3.0, plyer, requests, urllib3, certifi, openssl

# (list) Directories to exclude
source.exclude_dirs = venv, .venv, bin, .buildozer, build_env

# (int) Target Android API (Stable production target)
android.api = 33

# (int) Minimum API your APK will support (Android 8.0+)
android.minapi = 24

# (str) Android NDK directory 
android.ndk_path = /home/bilya/.buildozer/android/platform/android-ndk-r25c

# (str) Android NDK version (Fully supported online archival link)
android.ndk = r25c

# (list) Android permissions needed for camera and networking
android.permissions = INTERNET, CAMERA, READ_EXTERNAL_STORAGE, WRITE_EXTERNAL_STORAGE

# (str) Supported orientations
orientation = portrait