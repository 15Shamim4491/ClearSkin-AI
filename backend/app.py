"""
AI-Powered Acne Detection & Skin Analysis - Backend
----------------------------------------------------
Flask + OpenCV backend that:
 1. Receives an uploaded face/skin photo
 2. Detects the skin region (HSV + YCrCb skin color masking)
 3. Finds reddish/dark acne-like spots inside the skin region
    using color-space thresholding + contour analysis
 4. Scores severity (Mild / Moderate / Severe) based on spot
    count and total affected area
 5. Returns JSON: spot count, severity, bounding boxes, tips

Public-ready version: this file also serves the frontend page at "/",
so one Cloudflare Tunnel link works for both the website and the API.
"""

import base64
import os

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from PIL import Image

# --------------------------------------------------------------------------
# Frontend settings
# --------------------------------------------------------------------------
# Apnar main page er file name ekhane likhun (frontend folder-er bhitore):
MAIN_PAGE = "project (13).html"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(os.path.dirname(BASE_DIR), "frontend")
if not os.path.isdir(FRONTEND_DIR):
    FRONTEND_DIR = BASE_DIR  # fallback

app = Flask(__name__)
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = 15 * 1024 * 1024  # max 15 MB upload

MAX_IMAGE_DIM = 900  # resize large photos for speed/consistency


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def read_image_from_request(file_storage) -> np.ndarray:
    """Read an uploaded file into an OpenCV BGR image, resized for consistency."""
    pil_img = Image.open(file_storage.stream).convert("RGB")
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    h, w = img.shape[:2]
    scale = MAX_IMAGE_DIM / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    return img


def get_skin_mask(img_bgr: np.ndarray) -> np.ndarray:
    """Combine HSV + YCrCb thresholds to isolate skin-colored regions."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)

    lower_hsv = np.array([0, 15, 40], dtype=np.uint8)
    upper_hsv = np.array([25, 170, 255], dtype=np.uint8)
    mask_hsv = cv2.inRange(hsv, lower_hsv, upper_hsv)

    lower_ycrcb = np.array([0, 135, 85], dtype=np.uint8)
    upper_ycrcb = np.array([255, 180, 135], dtype=np.uint8)
    mask_ycrcb = cv2.inRange(ycrcb, lower_ycrcb, upper_ycrcb)

    skin_mask = cv2.bitwise_and(mask_hsv, mask_ycrcb)

    kernel = np.ones((5, 5), np.uint8)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return skin_mask


def detect_acne_spots(img_bgr: np.ndarray, skin_mask: np.ndarray):
    """
    Find acne-like blobs: reddish/inflamed or darker spots that stand out
    from the surrounding average skin tone, restricted to the skin region.
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    # Only look inside skin area
    skin_pixels = a_channel[skin_mask > 0]
    if skin_pixels.size == 0:
        return [], skin_mask

    mean_a = float(np.mean(skin_pixels))
    std_a = float(np.std(skin_pixels)) if np.std(skin_pixels) > 0 else 1.0

    # Acne spots push the 'a' (green-red) channel higher than average skin redness
    threshold = mean_a + max(6.0, 0.8 * std_a)
    _, redness_mask = cv2.threshold(a_channel, threshold, 255, cv2.THRESH_BINARY)
    redness_mask = cv2.bitwise_and(redness_mask, redness_mask, mask=skin_mask)

    kernel = np.ones((3, 3), np.uint8)
    redness_mask = cv2.morphologyEx(redness_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    redness_mask = cv2.dilate(redness_mask, kernel, iterations=1)

    contours, _ = cv2.findContours(redness_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    img_area = img_bgr.shape[0] * img_bgr.shape[1]
    min_area = max(6, img_area * 0.00006)   # ignore tiny noise
    max_area = img_area * 0.02              # ignore huge blobs (shadows/hair/background)

    boxes = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area or area > max_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        aspect = w / float(h) if h else 0
        if aspect < 0.25 or aspect > 4.0:  # skip long thin false-positives
            continue
        boxes.append({"x": int(x), "y": int(y), "w": int(w), "h": int(h), "area": float(area)})

    return boxes, redness_mask


def classify_severity(spot_count: int, img_area: int, total_spot_area: float):
    affected_pct = (total_spot_area / img_area) * 100 if img_area else 0

    if spot_count <= 5 and affected_pct < 0.4:
        severity = "Mild"
    elif spot_count <= 15 and affected_pct < 1.5:
        severity = "Moderate"
    else:
        severity = "Severe"

    tips = {
        "Mild": [
            "দিনে দুইবার মৃদু, সালফেট-মুক্ত ফেসওয়াশ দিয়ে মুখ পরিষ্কার রাখো।",
            "অতিরিক্ত মুখ স্পর্শ করা বা ব্রণ খুঁটে ফেলা এড়িয়ে চলো।",
            "নন-কমেডোজেনিক (non-comedogenic) ময়েশ্চারাইজার ব্যবহার করো।",
            "দিনে সানস্ক্রিন লাগানোর অভ্যাস করো।",
        ],
        "Moderate": [
            "Salicylic Acid বা Benzoyl Peroxide যুক্ত স্পট ট্রিটমেন্ট ব্যবহার করে দেখতে পারো।",
            "বালিশের কভার ও তোয়ালে নিয়মিত পরিবর্তন করো।",
            "তেল-জাতীয় (oily) খাবার ও অতিরিক্ত মিষ্টি কম খাওয়ার চেষ্টা করো।",
            "উন্নতি না হলে একজন ডার্মাটোলজিস্টের পরামর্শ নেওয়া ভালো।",
        ],
        "Severe": [
            "যত দ্রুত সম্ভব একজন চর্মরোগ বিশেষজ্ঞ (Dermatologist) এর সাথে পরামর্শ করো — এটা সবচেয়ে গুরুত্বপূর্ণ।",
            "নিজে থেকে কড়া কোনো ওষুধ বা ব্রণ চেপে বের করার চেষ্টা করো না, দাগ পড়ে যেতে পারে।",
            "ডাক্তারের পরামর্শ ছাড়া strong একনে ওষুধ (retinoid/antibiotic) ব্যবহার করো না।",
            "মুখ পরিষ্কার রাখা এবং হাত না লাগানোর অভ্যাস বজায় রাখো, চিকিৎসার পাশাপাশি।",
        ],
    }

    return severity, round(affected_pct, 2), tips[severity]


def encode_mask_preview(redness_mask: np.ndarray) -> str:
    """Return a small base64 PNG of the detected-spots mask (optional visual aid)."""
    ok, buf = cv2.imencode(".png", redness_mask)
    if not ok:
        return ""
    return "data:image/png;base64," + base64.b64encode(buf).decode("utf-8")


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------

@app.route("/", methods=["GET"])
def home():
    """Serve the website itself, so the public link opens the page directly."""
    page_path = os.path.join(FRONTEND_DIR, MAIN_PAGE)
    if os.path.isfile(page_path):
        return send_from_directory(FRONTEND_DIR, MAIN_PAGE)

    # Main page paoa jayni: available html file gulor list dekhai
    try:
        html_files = sorted(f for f in os.listdir(FRONTEND_DIR) if f.lower().endswith(".html"))
    except OSError:
        html_files = []

    items = "".join(f'<li><a href="/{f}">{f}</a></li>' for f in html_files) or "<li>Kono .html file paoa jayni</li>"
    return (
        f"<h2>'{MAIN_PAGE}' file ta paoa jayni</h2>"
        f"<p>Folder: {FRONTEND_DIR}</p>"
        f"<p>Niche theke ekta khule dekhun, tarpor app.py te MAIN_PAGE e sei naam likhun:</p>"
        f"<ul>{items}</ul>"
    ), 404


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/<path:filename>", methods=["GET"])
def frontend_files(filename):
    """Serve other frontend files (other pages, css, js, images)."""
    return send_from_directory(FRONTEND_DIR, filename)


@app.route("/predict", methods=["POST"])
def analyze():
    if "image" not in request.files:
        return jsonify({"error": "No image file uploaded. Use field name 'image'."}), 400

    file_storage = request.files["image"]
    if file_storage.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    try:
        img = read_image_from_request(file_storage)
    except Exception as exc:
        return jsonify({"error": f"Could not read image: {exc}"}), 400

    skin_mask = get_skin_mask(img)
    if cv2.countNonZero(skin_mask) < 0.03 * img.shape[0] * img.shape[1]:
        return jsonify({
            "error": "কোনো স্কিন/ফেস ডিটেক্ট করা যায়নি। দয়া করে মুখ স্পষ্টভাবে দেখা যায় এমন একটি ছবি দিন।"
        }), 422

    boxes, redness_mask = detect_acne_spots(img, skin_mask)
    total_spot_area = sum(b["area"] for b in boxes)
    img_area = img.shape[0] * img.shape[1]

    severity, affected_pct, tips = classify_severity(len(boxes), img_area, total_spot_area)

    return jsonify({
        "spot_count": len(boxes),
        "severity": severity,
        "affected_area_percent": affected_pct,
        "boxes": boxes,               # [{x, y, w, h, area}, ...] in resized-image pixel coords
        "image_width": img.shape[1],
        "image_height": img.shape[0],
        "tips": tips,
        "mask_preview": encode_mask_preview(redness_mask),  # optional debug/visual overlay
    })


if __name__ == "__main__":
    # debug=False is important: with debug=True a public link would expose
    # Flask's interactive debugger (anyone could run code on your PC).
    app.run(host="0.0.0.0", port=5000, debug=False)