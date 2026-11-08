import json
from pathlib import Path
import threading
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import urlopen

import pytest

from cropvolume.dashboard import Monitor, make_handler, path_key, count_written
from cropvolume.io import atomic_json
from cropvolume.profiling import span


def test_partial_capture_and_corrupt_state_are_visible_without_writes(tmp_path):
    root = tmp_path / 'dataset'
    sample = root / 'samples' / 'in-progress'
    (sample / 'images').mkdir(parents=True)
    atomic_json(sample / 'capture.json', {'frames': [{'image':'images/one.png'}, {'image':'images/two.png'}]})
    (sample / 'images/one.png').write_bytes(b'not-sealed-yet')
    (sample / 'worker.json').write_text('{partial write')
    before = {p: p.read_bytes() for p in root.rglob('*') if p.is_file()}
    data = Monitor([root]).snapshot()
    assert data['jobs'][0]['capture'] == {'done': 1, 'total': 2}
    assert data['jobs'][0]['status'] == 'capturing'
    assert data['summary']['accepted'] == 0 and data['warnings']
    assert before == {p:p.read_bytes() for p in before}


def test_dashboard_uses_real_receipts_and_does_not_invent_historical_stage_times(tmp_path):
    root = tmp_path / 'dataset'
    sample = root / 'samples' / 'done'
    sample.mkdir(parents=True)
    atomic_json(sample / 'COMPLETE.json', {'sample_id':'done'})
    atomic_json(sample / 'metrics.json', {'state':'complete','elapsed_s':71.94,'gaussians':100000,
        'test':{'mean_psnr_db':19.49,'mean_ssim':.682}})
    data = Monitor([root]).snapshot()
    assert data['summary']['accepted'] == 1
    assert data['jobs'][0]['historical_fit_s'] == 71.94
    assert data['timings'] == []


def test_scoped_profiles_attach_to_samples_even_when_stored_at_dataset_level(tmp_path):
    root = tmp_path / 'dataset'
    sample = root / 'samples' / 'sample-a'
    sample.mkdir(parents=True)
    with span('run', root=root, sample_id=sample.name):
        with span('feature_matching'):
            pass
    data = Monitor([root]).snapshot()
    assert len(data['jobs'][0]['profile_ids']) == 2
    assert all(p['job_id'] == data['jobs'][0]['id'] for p in data['profiles'])


def test_windows_profile_roots_normalize_to_wsl_paths():
    assert path_key(r'E:\CropVolumeDataset\pilot') == '/mnt/e/CropVolumeDataset/pilot'
    assert path_key(r'\\wsl.localhost\Ubuntu\home\bren\simProject') == '/home/bren/simProject'


def test_frame_counts_use_actual_files_and_exclude_unsafe_paths(tmp_path):
    root = tmp_path / 'sample'
    (root / 'images/sub').mkdir(parents=True)
    (root / 'images/one.png').write_bytes(b'image')
    (root / 'images/sub/two.png').write_bytes(b'image')
    (root / 'images/folder.png').mkdir()
    outside = tmp_path / 'outside.png'
    outside.write_bytes(b'image')
    (root / 'images/link.png').symlink_to(outside)
    frames = [{'image':p} for p in ['images/one.png','images/sub/two.png','images/missing.png',
        'images/folder.png','images/link.png','../outside.png','/outside.png']]
    assert count_written(root,frames) == 2


def test_live_http_endpoints_export_and_do_not_serve_arbitrary_files(tmp_path):
    root = tmp_path / 'dataset'
    (root / 'samples').mkdir(parents=True)
    monitor = Monitor([root])
    try:
        server = ThreadingHTTPServer(('127.0.0.1',0), make_handler(monitor))
    except PermissionError:
        pytest.skip('Local sockets are restricted; run this HTTP check outside the sandbox')
    thread = threading.Thread(target=server.serve_forever,daemon=True)
    thread.start()
    base = f'http://127.0.0.1:{server.server_port}'
    try:
        with urlopen(base+'/api/status') as response:
            assert json.load(response)['schema'] == 'cropvolume-dashboard-v1'
        with urlopen(base+'/') as response:
            assert b'Dataset generator' in response.read()
        with urlopen(base+'/api/timings.csv') as response:
            assert response.headers['Content-Disposition']
            assert response.read().startswith(b'stage,calls')
        for route in ('/../../etc/passwd','/api/preview/../../etc/passwd','/api/status/extra'):
            with pytest.raises(HTTPError) as error:
                urlopen(base+route)
            assert error.value.code == 404
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
