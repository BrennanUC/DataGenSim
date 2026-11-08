import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from http.server import ThreadingHTTPServer
import threading
from urllib.error import HTTPError
from urllib.request import Request,urlopen

import pytest

from cropvolume.dashboard import Monitor,make_handler
from cropvolume.generation import GenerationManager,DEFAULTS,pipeline_config
from cropvolume.io import atomic_json,read_json
from cropvolume.profiling import profile_directory,_process_identity
from cropvolume.run_control import stop_processes,process_info


def managed_run(parent,name='random-field-controls-test',state='stopped'):
    root=parent/name;(root/'samples/sample-a').mkdir(parents=True)
    atomic_json(root/'generation.json',{'schema':'cropvolume-generation-v3','settings':DEFAULTS,
        'config':pipeline_config({}),'runtime_root':str(parent),'catalog':'unused','engine_root':'unused','created_unix':time.time()})
    atomic_json(root/'generation_status.json',{'state':state})
    atomic_json(root/'reservations.json',{'sample-a':1000})
    return root


def test_clear_temporary_preserves_photos_labels_pairs_and_resets_stage_references(tmp_path):
    root=managed_run(tmp_path,state='complete');sample=root/'samples/sample-a'
    for name in ['images/one.png','labels/volume.json','splat.ply','metadata.json','COMPLETE.json']:
        path=sample/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'published immutable file')
    (sample/'work/fit').mkdir(parents=True);(sample/'work/fit/checkpoint.pt').write_bytes(b'checkpoint')
    (sample/'copy.tmp').write_bytes(b'partial copy')
    atomic_json(sample/'worker.json',{'state':'complete','attempts':2,'prepared':'work/prep','training_output':'work/fit'})
    profiles=profile_directory(root);profiles.mkdir(parents=True);(profiles/'span.json').write_text('{}')
    before={name:(sample/name).read_bytes() for name in ['images/one.png','labels/volume.json','splat.ply','metadata.json','COMPLETE.json']}
    manager=GenerationManager(tmp_path)
    preview=manager.cleanup_preview(root.name)
    assert preview['files']==2 and preview['published_pairs']==1
    manager.clear_temporary(root.name,root.name)
    assert not (sample/'work').exists() and not (sample/'copy.tmp').exists()
    assert before=={name:(sample/name).read_bytes() for name in before}
    assert profiles.exists() and read_json(root/'generation_status.json')['state']=='complete'
    state=read_json(sample/'worker.json');assert state['state']=='complete' and 'prepared' not in state
    assert read_json(root/'reservations.json')=={}


def test_clear_temporary_resets_unfinished_preparation_for_rebuild(tmp_path):
    root=managed_run(tmp_path);sample=root/'samples/sample-a'
    atomic_json(sample/'CAPTURE_COMPLETE.json',{})
    atomic_json(sample/'worker.json',{'state':'failed','attempts':3,'prepared':'work/partial'})
    (sample/'work/partial').mkdir(parents=True)
    manager=GenerationManager(tmp_path);manager.clear_temporary(root.name,root.name)
    state=read_json(sample/'worker.json')
    assert state['state']=='ready' and state['attempts']==0 and 'prepared' not in state


def test_full_deletion_removes_only_run_and_its_sidecars(tmp_path):
    root=managed_run(tmp_path);other=managed_run(tmp_path,'random-field-other')
    assets=tmp_path/'assets';assets.mkdir();(assets/'catalog.json').write_text('shared')
    profiles=profile_directory(root);profiles.mkdir(parents=True);(profiles/'span.json').write_text('{}')
    other_profiles=profile_directory(other);other_profiles.mkdir(parents=True)
    manager=GenerationManager(tmp_path)
    preview=manager.cleanup_preview(root.name,'run');assert preview['files']>=4
    manager.delete(root.name,root.name)
    assert not root.exists() and not profiles.exists()
    assert other.exists() and other_profiles.exists() and (assets/'catalog.json').read_text()=='shared'


def test_destructive_actions_require_exact_confirmation_and_managed_path(tmp_path):
    root=managed_run(tmp_path);manager=GenerationManager(tmp_path)
    for action in (manager.delete,manager.clear_temporary):
        with pytest.raises(ValueError,match='Confirm'):action(root.name,'different-run')
        with pytest.raises(ValueError):action('../assets','../assets')
    assert root.exists()


def test_cleanup_unlinks_work_symlink_without_following_it(tmp_path):
    root=managed_run(tmp_path);outside=tmp_path/'outside';outside.mkdir();(outside/'keep').write_text('keep')
    (root/'samples/sample-a/work').symlink_to(outside,target_is_directory=True)
    GenerationManager(tmp_path).clear_temporary(root.name,root.name)
    assert (outside/'keep').read_text()=='keep' and not (root/'samples/sample-a/work').exists()


@pytest.mark.parametrize('path',['samples','locks','generation_status.json'])
def test_symlinked_managed_paths_cannot_redirect_cleanup(tmp_path,path):
    root=managed_run(tmp_path);outside=tmp_path/'outside';outside.mkdir()
    target=root/path
    if target.is_dir():
        import shutil;shutil.rmtree(target)
    else:target.unlink(missing_ok=True)
    target.symlink_to(outside,target_is_directory=True)
    with pytest.raises(ValueError,match='symlinks'):GenerationManager(tmp_path).delete(root.name,root.name)
    assert root.exists() and outside.exists()


def test_run_directory_symlink_cannot_delete_another_run(tmp_path):
    actual=managed_run(tmp_path);link=tmp_path/'random-field-linked';link.symlink_to(actual,target_is_directory=True)
    with pytest.raises(ValueError):GenerationManager(tmp_path).delete(link.name,link.name)
    assert actual.exists()


def test_live_external_lease_prevents_deletion(tmp_path):
    root=managed_run(tmp_path)
    atomic_json(root/'locks/other/owner.json',{'pid':os.getpid(),'host':socket.gethostname()})
    with pytest.raises(RuntimeError,match='live process'):GenerationManager(tmp_path).delete(root.name,root.name)
    assert root.exists()


def test_immediate_stop_terminates_verified_worker_and_descendant(tmp_path):
    root=managed_run(tmp_path,state='running')
    child_file=root/'child.pid'
    code="import subprocess,sys,time,signal; from pathlib import Path; signal.signal(signal.SIGINT,signal.SIG_IGN); child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(120)']); Path(sys.argv[1]).write_text(str(child.pid)); time.sleep(120)"
    process=subprocess.Popen([sys.executable,'-c',code,str(child_file),'--job',str(root/'generation.json')],
        start_new_session=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    try:
        deadline=time.monotonic()+3
        while not child_file.exists() and time.monotonic()<deadline:time.sleep(.02)
        assert child_file.exists()
        state={'state':'running','pid':process.pid,'process_identity':_process_identity(process.pid)}
        stopped=stop_processes(root,state,timeout=.1)
        process.wait(timeout=3)
        child=process_info(int(child_file.read_text()))
        assert not child or child['state']=='Z'
        assert process.pid in stopped['worker_pids'] and (root/'generation.json').exists()
    finally:
        if process.poll() is None:os.killpg(process.pid,signal.SIGKILL);process.wait()


def test_pid_reuse_does_not_signal_an_unrelated_process(tmp_path,monkeypatch):
    root=managed_run(tmp_path)
    import cropvolume.run_control as control
    called=[];monkeypatch.setattr(control.os,'kill',lambda *args:called.append(args))
    stopped=stop_processes(root,{'pid':os.getpid(),'process_identity':'not-this-process'})
    assert called==[] and stopped['worker_pids']==[]


def test_stopped_run_does_not_show_unfinished_windows_spans_as_running(tmp_path):
    root=managed_run(tmp_path);sample=root/'samples/sample-a'
    atomic_json(sample/'producer.json',{'state':'capturing'})
    folder=profile_directory(sample)
    atomic_json(folder/'native.json',{'schema':'cropvolume-profile-span-v1','span_id':'native','stage':'render_views',
        'status':'running','metadata':{},'job_root':str(sample),'platform':'nt','host':'windows','started_unix':time.time()})
    data=Monitor([tmp_path]).snapshot()
    assert data['jobs'][0]['status']=='interrupted'
    assert data['profiles'][0]['status']=='interrupted' and data['profiles'][0]['elapsed_s'] is None
    assert data['summary']['active']==0


def test_new_manual_activity_after_run_finished_is_still_visible(tmp_path):
    root=managed_run(tmp_path)
    atomic_json(root/'generation_status.json',{'state':'complete','updated_unix':time.time()-10})
    folder=profile_directory(root)
    atomic_json(folder/'verify.json',{'schema':'cropvolume-profile-span-v1','span_id':'new-verify','stage':'verification',
        'status':'running','metadata':{},'job_root':str(root),'platform':os.name,'host':socket.gethostname(),
        'pid':os.getpid(),'process_identity':_process_identity(os.getpid()),'started_unix':time.time()})
    data=Monitor([tmp_path]).snapshot()
    assert data['profiles'][0]['status']=='running'


def test_published_pair_stays_prepared_in_monitor_after_work_cache_is_cleared(tmp_path):
    root=managed_run(tmp_path,state='complete');sample=root/'samples/sample-a'
    atomic_json(sample/'COMPLETE.json',{'sample_id':'sample-a'})
    manager=GenerationManager(tmp_path);manager.clear_temporary(root.name,root.name)
    data=Monitor([tmp_path]).snapshot()
    assert data['jobs'][0]['status']=='complete' and data['jobs'][0]['phases']['preparation'] is True


def test_http_delete_needs_session_and_confirmation(tmp_path):
    root=managed_run(tmp_path);manager=GenerationManager(tmp_path)
    try:server=ThreadingHTTPServer(('127.0.0.1',0),make_handler(Monitor([tmp_path]),manager))
    except PermissionError:pytest.skip('Local sockets restricted')
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    base=f'http://127.0.0.1:{server.server_port}/api/generation/{root.name}'
    def post(action,body,token=None):
        headers={'Content-Type':'application/json'}
        if token:headers['X-CropVolume-Token']=token
        return urlopen(Request(base+'/'+action,json.dumps(body).encode(),headers))
    try:
        with pytest.raises(HTTPError) as error:post('delete',{'confirm_run_id':root.name})
        assert error.value.code==403 and root.exists()
        with pytest.raises(HTTPError) as error:post('delete',{'confirm_run_id':'wrong'},manager.token)
        assert error.value.code==400 and root.exists()
        with post('cleanup-preview',{'scope':'run'},manager.token) as response:
            assert json.load(response)['scope']=='run'
        assert root.exists()
        with post('delete',{'confirm_run_id':root.name},manager.token) as response:
            assert response.status==202
        assert not root.exists()
    finally:server.shutdown();server.server_close();thread.join(timeout=2)
