from pathlib import Path
import os


# this size is required for embedding
FACE_PIC_SIZE = 160

EMBEDDING_SIZE = 512

PRETREINED_MODEL_DIR = os.path.join(str(Path.home()), 'pretrained_models')

UNKNOWN_CLASS = "unknown"