import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest.mock import Mock
from rl_workbench.microduck import MicroduckManager
from rl_workbench.server import create_server
from rl_workbench.training import atomic_json

RUN_ID = 'abcdef123456'


def record(root, status='completed'):
    folder = Path(root) / RUN_ID
    folder.mkdir(exist_ok=True)
    atomic_json(folder / 'run.json', {'id': RUN_ID, 'status': status, 'created_at': '2026-10-09T00:00:00+00:00', 'config': {}, 'iteration': 0})
    (folder / 'official').mkdir(exist_ok=True)
    (folder / 'official' / 'model_0.pt').write_bytes(b'fixture')
    return folder


class RunDeletionTests(unittest.TestCase):
    def test_removes_only_selected_run_and_not_symlink_targets(self):
        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as outside:
            manager = MicroduckManager(root)
            folder = record(root)
            keeper = Path(root) / 'other.txt'; keeper.write_text('keep')
            external = Path(outside) / 'model.pt'; external.write_text('keep')
            (folder / 'linked').symlink_to(outside, target_is_directory=True)
            self.assertEqual(manager.delete(RUN_ID), {'deleted': RUN_ID})
            self.assertFalse(folder.exists())
            self.assertEqual(keeper.read_text(), 'keep')
            self.assertEqual(external.read_text(), 'keep')
            with self.assertRaises(KeyError): manager.delete(RUN_ID)
            folder.symlink_to(outside, target_is_directory=True)
            with self.assertRaises(ValueError): manager.delete(RUN_ID)
            for value in ['../outside', '', '/', None]:
                with self.assertRaises(ValueError): manager.delete(value)

    def test_all_active_states_and_live_process_are_protected(self):
        with tempfile.TemporaryDirectory() as root:
            manager = MicroduckManager(root)
            for status in ['starting','running','paused','pausing','stopping']:
                folder = record(root, status)
                with self.assertRaises(RuntimeError): manager.delete(RUN_ID)
                self.assertTrue(folder.exists())
            record(root, 'completed')
            manager.current_id = RUN_ID
            manager.process = Mock(); manager.process.poll.return_value = None
            with self.assertRaises(RuntimeError): manager.delete(RUN_ID)
            manager.process.poll.return_value = 0
            manager.delete(RUN_ID)
            self.assertIsNone(manager.current_id)

    def test_http_delete_protection_and_missing_records(self):
        with tempfile.TemporaryDirectory() as root:
            server = create_server(0, root)
            thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
            base = f'http://127.0.0.1:{server.server_port}'
            folder = record(server.microduck.root)
            endpoint = base + f'/api/microduck/runs/{RUN_ID}/delete'
            def post(origin=None):
                headers = {'Content-Type': 'application/json'}
                if origin: headers['Origin'] = origin
                return urllib.request.urlopen(urllib.request.Request(endpoint, data=b'{}', headers=headers))
            try:
                with self.assertRaises(urllib.error.HTTPError) as error: post('https://foreign.example')
                self.assertEqual(error.exception.code, 403)
                self.assertTrue(folder.exists())
                record(server.microduck.root, 'paused')
                with self.assertRaises(urllib.error.HTTPError) as error: post()
                self.assertEqual(error.exception.code, 409)
                record(server.microduck.root, 'failed')
                with post() as response: self.assertEqual(json.load(response), {'deleted': RUN_ID})
                with urllib.request.urlopen(base+'/api/microduck/runs') as response: self.assertEqual(json.load(response), [])
                for path in ['', '/export', '/latest', '/download/model_0.pt']:
                    with self.assertRaises(urllib.error.HTTPError) as error:
                        urllib.request.urlopen(base+f'/api/microduck/runs/{RUN_ID}'+path)
                    self.assertEqual(error.exception.code, 404)
                with self.assertRaises(urllib.error.HTTPError) as error: post()
                self.assertEqual(error.exception.code, 404)
            finally:
                server.shutdown(); server.server_close()
