ACTIONS = [
    "hello",
    "yes",
    "no",
    "thank you",
    "good morning",
]

LANGUAGES = {
    "Hindi (hi)": "hi",
    "Tamil (ta)": "ta",
    "Telugu (te)": "te",
    "Bengali (bn)": "bn",
    "English (en)": "en",
}

DEFAULT_LANG = "hi"

SEQ_LEN = 30
FEATURES_LEN = 1662  # face(468*3) + pose(33*4) + hands(21*3*2)

MODEL_DIR = "models"
MODEL_PATH = f"{MODEL_DIR}/signlink_lstm.h5"
LABELS_PATH = f"{MODEL_DIR}/labels.json"

DATA_DIR = "data"
