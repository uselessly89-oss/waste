import cv2


def assess_frame(frame, min_brightness=25, max_brightness=235, min_blur_variance=30):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean())
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness_ok = min_brightness <= brightness <= max_brightness
    sharp_ok = sharpness >= min_blur_variance
    return {
        "ok": brightness_ok and sharp_ok,
        "brightness": brightness,
        "sharpness": sharpness,
        "reason": "ok" if brightness_ok and sharp_ok else "poor_frame_quality",
    }
