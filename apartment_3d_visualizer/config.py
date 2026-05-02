import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
FURNITURE_DIR = os.path.join(ASSETS_DIR, "furniture")
MATERIALS_DIR = os.path.join(ASSETS_DIR, "materials")
PROJECTS_DIR = os.path.join(BASE_DIR, "projects")
RENDERS_DIR = os.path.join(BASE_DIR, "renders")
CATALOG_FILE = os.path.join(ASSETS_DIR, "catalog.json")

DEFAULT_WALL_HEIGHT = 2.7
DEFAULT_WALL_THICKNESS = 0.2

IMAGE_PROCESSING = {
    "blur_kernel": 5,
    "min_contour_area": 500,
    "approx_epsilon_factor": 0.02,
    "morph_kernel_size": 3,
}

# Масштаб плана (пикселей на метр).
# По умолчанию выбран диапазон, типичный для онлайн-планировщиков; при AUTO_SCALE_PIXELS_PER_METER=None
# масштаб подбирается автоматически по ожидаемым габаритам квартиры.
DEFAULT_PIXELS_PER_METER = 70.0
AUTO_SCALE_PIXELS_PER_METER = None  # None = авто, число = фиксированное значение px/м

# Ожидаемые габариты плана (в метрах). Используются для автоподбора масштаба.
EXPECTED_PLAN_WIDTH_METERS  = (5.0, 30.0)   # ширина плана, м
EXPECTED_PLAN_HEIGHT_METERS = (3.0, 25.0)   # высота плана, м

RENDER_SETTINGS = {
    "engine": "CYCLES",
    "samples": 128,
    "resolution_x": 1920,
    "resolution_y": 1080,
}

CAMERA_PRESETS = {
    "top": {
        "location": (0, 0, 15),
        "rotation": (0, 0, 0),
        "type": "ORTHO",
        "ortho_scale": 20,
    },
    "perspective": {
        "location": (10, -10, 8),
        "rotation": (1.0, 0, 0.8),
        "type": "PERSP",
        "focal_length": 35,
    },
    "first_person": {
        "location": (0, 0, 1.7),
        "rotation": (1.5708, 0, 0),
        "type": "PERSP",
        "focal_length": 28,
    },
}

DEFAULT_MATERIALS = {
    "wall":    {"color": (0.9,  0.9,  0.85, 1.0), "roughness": 0.8},
    "floor":   {"color": (0.6,  0.5,  0.4,  1.0), "roughness": 0.4},
    "ceiling": {"color": (1.0,  1.0,  1.0,  1.0), "roughness": 0.9},
}

for d in [ASSETS_DIR, FURNITURE_DIR, MATERIALS_DIR, PROJECTS_DIR, RENDERS_DIR]:
    os.makedirs(d, exist_ok=True)
