"""A bounded, atomically replaced preview, independent of diagnostic history."""
import base64
import io
import json
import time
from pathlib import Path


def write_preview(directory, image, env_step):
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=75)
    frame = {"env_step": int(env_step), "captured_at": time.time(),
             "image": "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")}
    target = Path(directory) / "latest.json"
    pending = target.with_suffix(".tmp")
    pending.write_text(json.dumps(frame), encoding="utf-8")
    pending.replace(target)
