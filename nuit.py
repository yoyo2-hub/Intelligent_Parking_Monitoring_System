import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

# ========== CONFIGURATION ==========
INPUT_VIDEO = "video.mp4"
OUTPUT_VIDEO = "video_night_ultra_realistic.mp4"

# Profils prédéfinis (choisir un)
NIGHT_PROFILE = "realistic"  # Options: "realistic", "deep_night", "twilight", "security_cam"

PROFILES = {
    "realistic": {
        "brightness": -65,
        "contrast": 0.75,
        "gamma": 0.65,
        "blue_shift": 15,
        "saturation": 0.5,
        "vignette": 0.3,
        "bloom_threshold": 200,
        "bloom_intensity": 1.5,
        "noise": 3,
        "shadow_enhance": 1.4
    },
    "deep_night": {
        "brightness": -85,
        "contrast": 0.6,
        "gamma": 0.5,
        "blue_shift": 25,
        "saturation": 0.3,
        "vignette": 0.5,
        "bloom_threshold": 180,
        "bloom_intensity": 2.0,
        "noise": 6,
        "shadow_enhance": 1.6
    },
    "twilight": {
        "brightness": -45,
        "contrast": 0.85,
        "gamma": 0.75,
        "blue_shift": 10,
        "saturation": 0.65,
        "vignette": 0.2,
        "bloom_threshold": 210,
        "bloom_intensity": 1.2,
        "noise": 2,
        "shadow_enhance": 1.2
    },
    "security_cam": {
        "brightness": -70,
        "contrast": 0.9,
        "gamma": 0.6,
        "blue_shift": 5,
        "saturation": 0.2,
        "vignette": 0.4,
        "bloom_threshold": 190,
        "bloom_intensity": 1.8,
        "noise": 8,
        "shadow_enhance": 1.5
    }
}

# ========== FONCTIONS AVANCÉES ==========

def apply_gamma_correction(image, gamma=1.0):
    """Correction gamma avec lookup table optimisée"""
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 
                      for i in range(256)]).astype("uint8")
    return cv2.LUT(image, table)

def enhance_shadows(image, factor=1.5):
    """Accentue les ombres pour effet nocturne naturel"""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    l_channel = lab[:, :, 0]
    
    # Compression non-linéaire des zones sombres
    l_channel = np.where(l_channel < 128, 
                         l_channel / factor, 
                         l_channel)
    
    lab[:, :, 0] = np.clip(l_channel, 0, 255)
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

def apply_blue_temperature_shift(image, intensity=15):
    """Simule la température de couleur froide nocturne"""
    shifted = image.astype(np.float32)
    
    # Augmente bleu, réduit rouge/jaune
    shifted[:, :, 0] += intensity          # Plus de bleu
    shifted[:, :, 1] -= intensity * 0.3    # Moins de vert
    shifted[:, :, 2] -= intensity * 0.5    # Moins de rouge
    
    return np.clip(shifted, 0, 255).astype(np.uint8)

def adjust_saturation_adaptive(image, factor=0.5):
    """Désaturation adaptative (préserve les zones lumineuses)"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    v_channel = hsv[:, :, 2]
    
    # Les zones sombres perdent plus de saturation
    saturation_map = np.clip(v_channel / 255.0, 0.3, 1.0)
    hsv[:, :, 1] *= (factor * saturation_map)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
    
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

def create_vignette(image, intensity=0.3):
    """Crée un vignettage naturel (assombrissement des bords)"""
    h, w = image.shape[:2]
    
    # Masque radial gaussien
    center_x, center_y = w // 2, h // 2
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
    max_dist = np.sqrt(center_x**2 + center_y**2)
    
    vignette_mask = 1 - (dist / max_dist) * intensity
    vignette_mask = np.clip(vignette_mask, 0, 1)
    
    # Applique le masque
    vignetted = image.astype(np.float32)
    for i in range(3):
        vignetted[:, :, i] *= vignette_mask
    
    return np.clip(vignetted, 0, 255).astype(np.uint8)

def apply_light_bloom(image, threshold=200, intensity=1.5):
    """Simule le halo lumineux autour des sources de lumière"""
    # Extraction des zones très lumineuses
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bright_mask = np.where(gray > threshold, gray, 0).astype(np.float32)
    
    # Flou gaussien pour créer le halo
    bloom = cv2.GaussianBlur(bright_mask, (21, 21), 0)
    bloom = (bloom / 255.0) * intensity
    
    # Ajoute le bloom à l'image
    bloomed = image.astype(np.float32)
    for i in range(3):
        bloomed[:, :, i] += bloom * 255
    
    return np.clip(bloomed, 0, 255).astype(np.uint8)

def add_film_grain(image, intensity=5):
    """Ajoute un grain réaliste de type caméra nocturne"""
    # Bruit gaussien avec distribution non-uniforme
    noise = np.random.normal(0, intensity, image.shape).astype(np.float32)
    
    # Plus de bruit dans les zones sombres
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    noise_factor = 1 + (1 - gray / 255.0) * 0.5
    noise *= noise_factor[:, :, np.newaxis]
    
    noisy = image.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)

def apply_subtle_color_grading(image):
    """Color grading subtil type cinéma nocturne"""
    # Conversion en LAB pour manipulation perceptuelle
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    
    # Compression légère des hautes lumières
    l_channel = lab[:, :, 0]
    l_channel = np.where(l_channel > 180, 
                         180 + (l_channel - 180) * 0.6, 
                         l_channel)
    
    # Shift vers bleu-cyan dans les mid-tones
    a_channel = lab[:, :, 1] - 3
    b_channel = lab[:, :, 2] - 5
    
    lab[:, :, 0] = l_channel
    lab[:, :, 1] = np.clip(a_channel, 0, 255)
    lab[:, :, 2] = np.clip(b_channel, 0, 255)
    
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

# ========== PIPELINE PRINCIPAL ==========

def process_frame_to_night(frame, profile):
    """Pipeline complet de transformation nuit ultra-réaliste"""
    
    # 1. Ajustement de base (luminosité + contraste)
    night = cv2.convertScaleAbs(frame, alpha=profile["contrast"], 
                                beta=profile["brightness"])
    
    # 2. Correction gamma (assombrissement naturel)
    night = apply_gamma_correction(night, profile["gamma"])
    
    # 3. Accentuation des ombres
    night = enhance_shadows(night, profile["shadow_enhance"])
    
    # 4. Shift de température couleur (bleu nocturne)
    night = apply_blue_temperature_shift(night, profile["blue_shift"])
    
    # 5. Désaturation adaptative
    night = adjust_saturation_adaptive(night, profile["saturation"])
    
    # 6. Bloom sur sources lumineuses (lampadaires, phares)
    night = apply_light_bloom(night, profile["bloom_threshold"], 
                             profile["bloom_intensity"])
    
    # 7. Vignettage naturel
    night = create_vignette(night, profile["vignette"])
    
    # 8. Color grading cinématique
    night = apply_subtle_color_grading(night)
    
    # 9. Grain de film/bruit caméra
    if profile["noise"] > 0:
        night = add_film_grain(night, profile["noise"])
    
    return night

# ========== TRAITEMENT VIDÉO ==========

profile = PROFILES[NIGHT_PROFILE]

cap = cv2.VideoCapture(INPUT_VIDEO)
if not cap.isOpened():
    raise Exception(f"❌ Impossible d'ouvrir {INPUT_VIDEO}")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Encodeur H.264 haute qualité
fourcc = cv2.VideoWriter_fourcc(*"avc1")
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

print(f"🌙 Conversion ultra-réaliste en mode : {NIGHT_PROFILE.upper()}")
print(f"📐 {width}x{height} @ {fps} FPS | {total_frames} frames")
print("=" * 60)

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    
    # Appliquer le pipeline nocturne
    night_frame = process_frame_to_night(frame, profile)
    
    out.write(night_frame)
    
    # Barre de progression
    if frame_count % 30 == 0 or frame_count == total_frames:
        progress = (frame_count / total_frames) * 100
        bar_length = 40
        filled = int(bar_length * frame_count // total_frames)
        bar = "█" * filled + "░" * (bar_length - filled)
        print(f"\r⏳ [{bar}] {progress:.1f}% ({frame_count}/{total_frames})", end="")

print(f"\n\n✅ Vidéo nocturne ultra-réaliste créée : {OUTPUT_VIDEO}")
print(f"🎬 Profil utilisé : {NIGHT_PROFILE}")
print(f"📊 {frame_count} frames traitées avec succès")

cap.release()
out.release()