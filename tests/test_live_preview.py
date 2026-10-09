import base64
import io
import json
import tempfile
import threading
import unittest
import urllib.request
from pathlib import Path
from PIL import Image
from rl_workbench.live_preview import write_preview
from rl_workbench.server import create_server


class LivePreviewTests(unittest.TestCase):
    def test_replaces_frame_and_serves_without_training_history(self):
        with tempfile.TemporaryDirectory() as root:
            server = create_server(0, root)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                folder = Path(root) / 'microduck' / 'abcdef123456'
                folder.mkdir()
                for step in range(30):
                    write_preview(folder, Image.new('RGB', (16, 16), (step, 0, 0)), step)
                self.assertEqual([p.name for p in folder.iterdir()], ['latest.json'])
                url = f'http://127.0.0.1:{server.server_port}/api/microduck/runs/abcdef123456/latest'
                with urllib.request.urlopen(url) as response:
                    self.assertEqual(response.headers['Cache-Control'], 'no-store')
                    frame = json.load(response)
                self.assertEqual(frame['env_step'], 29)
                image = Image.open(io.BytesIO(base64.b64decode(frame['image'].split(',')[1])))
                self.assertEqual(image.size, (16, 16))
                self.assertEqual(image.format, 'JPEG')
            finally:
                server.shutdown()
                server.server_close()
