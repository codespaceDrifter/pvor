import os

def usb_path():
    media_path = f"/media/{os.getlogin()}"
    if os.path.exists(media_path):
        for item in os.listdir(media_path):
            usb_path = os.path.join(media_path, item)
            if os.path.ismount(usb_path):
                return usb_path
    return None