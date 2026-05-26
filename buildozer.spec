[app]
# (str) Title of your application
title = voiceOversea

# (str) Package name
package.name = voiceoversea

# (str) Package domain (needed for android/ios packaging)
package.domain = org.assistive

# (list) Source files to include (let's include py and assets)
source.include_exts = py,png,jpg,kv,atlas,json

# (list) Application requirements
requirements = python3, kivy==2.3.0, plyer, requests, urllib3, certifi, openssl

# (list) Directories to exclude
source.exclude_dirs = venv, .venv, bin, .buildozer

# (int) Target Android API
android.api = 33

# (int) Minimum API your APK will support
android.minapi = 24

# (list) Android permissions needed for camera and networking
android.permissions = INTERNET, CAMERA, READ_EXTERNAL_STORAGE, WRITE_EXTERNAL_STORAGE

# (str) Supported orientations
orientation = portrait