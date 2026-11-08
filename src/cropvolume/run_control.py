"""Stop only verified run processes and clean only that managed run's files."""
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import stat
import subprocess
import time

from .io import atomic_json, read_json
from .profiling import _process_identity, profile_directory


def process_info(pid):
    try:
        root=Path('/proc')/str(int(pid))
        fields=(root/'stat').read_text().rsplit(')',1)[1].split()
        return {'pid':int(pid),'state':fields[0],'group':int(fields[2]),'session':int(fields[3]),
                'identity':fields[19],'args':(root/'cmdline').read_bytes().split(b'\0')}
    except (OSError,ValueError,IndexError,TypeError):
        return None


def worker_matches(info,root,identity=None):
    if not info or info['state']=='Z' or (identity and info['identity']!=identity):
        return False
    args=info['args']
    try:
        index=args.index(b'--job')
        return args[index+1]==str(root/'generation.json').encode()
    except (ValueError,IndexError):
        return False


def live_run_workers(root,status):
    candidates={status.get('pid'):status.get('process_identity')}
    launch=root/'generation_process.json'
    if launch.exists():
        row=read_json(launch);candidates[row.get('pid')]=row.get('process_identity')
    return [info for pid,identity in candidates.items()
            if worker_matches(info:=process_info(pid),root,identity)]


def stop_windows_renderers(root):
    samples=[]
    for sample in (root/'samples').iterdir() if (root/'samples').exists() else []:
        if sample.is_symlink() or not sample.is_dir():continue
        record=sample/'unreal_process.json'
        if record.exists():
            try:
                if read_json(record).get('state')=='exited':continue
            except (OSError,ValueError):pass
        if record.exists() or ((sample/'unreal_capture.log').exists() and not (sample/'RENDER_FINISHED.json').exists()):
            samples.append(sample)
    if not samples:return []
    from .batch import windows_path
    powershell=shutil.which('powershell.exe')
    if not powershell:
        raise RuntimeError('Cannot verify/stop the Windows renderer. Keep the files until Windows process control is available.')
    script=Path(__file__).resolve().parents[2]/'tools/stop_unreal_capture.ps1'
    stopped=[]
    for sample in samples:
        result=subprocess.run([powershell,'-NoProfile','-NonInteractive','-ExecutionPolicy','Bypass',
            '-File',windows_path(script),'-Sample',windows_path(sample)],capture_output=True,text=True,timeout=20)
        if result.returncode:
            raise RuntimeError('Could not stop the run renderer: '+(result.stderr or result.stdout)[-1000:])
        try:stopped.extend(json.loads(result.stdout.strip()).get('stopped_pids',[]))
        except (ValueError,AttributeError):raise RuntimeError('Could not verify Windows renderer shutdown')
    return stopped


def stop_processes(root,status,timeout=4.):
    """Interrupt Python first, then terminate its verified detached session if needed."""
    # Validate all worker groups before signaling anything.
    workers=live_run_workers(root,status)
    groups=set()
    for info in workers:
        if info['group']!=info['session'] or info['group']==os.getpgrp():
            raise RuntimeError('Worker is not in a dedicated run session; refusing to stop unrelated processes')
        groups.add(info['group'])
    windows=stop_windows_renderers(root)
    for info in workers:
        if worker_matches(process_info(info['pid']),root,info['identity']):
            try:os.kill(info['pid'],signal.SIGINT)
            except ProcessLookupError:pass
    def members():
        found=[]
        for path in Path('/proc').iterdir():
            if not path.name.isdigit():continue
            info=process_info(path.name)
            if info and info['group'] in groups and info['session'] in groups and info['state']!='Z':found.append(info)
        return found
    deadline=time.monotonic()+timeout
    while members() and time.monotonic()<deadline:time.sleep(.1)
    for sig in (signal.SIGTERM,signal.SIGKILL):
        remaining=members()
        if not remaining:break
        for group in {info['group'] for info in remaining}:
            try:os.killpg(group,sig)
            except ProcessLookupError:pass
        deadline=time.monotonic()+2
        while members() and time.monotonic()<deadline:time.sleep(.1)
    if members():raise RuntimeError('Run processes are still active; files were not cleared')
    return {'worker_pids':[info['pid'] for info in workers],'windows_pids':windows,
            'memory_release':'Run processes exited; their RAM/VRAM can be reclaimed by the operating system/driver.'}


def assert_no_live_leases(root):
    for owner in (root/'locks').glob('*/owner.json'):
        row=read_json(owner)
        if row.get('host')!=socket.gethostname():
            raise RuntimeError('Another host holds a run lease; files were not cleared')
        info=process_info(row.get('pid'))
        if info and info['state']!='Z':
            raise RuntimeError('A live process still holds a run lease; files were not cleared')


def release_run_bookkeeping(root,runtime_root,stopped):
    assert_no_live_leases(root)
    if (root/'reservations.json').exists():atomic_json(root/'reservations.json',{})
    # Release only the shared lease owned by this worker, never another run's lease.
    expected=set(stopped['worker_pids'])
    runtime_root=Path(runtime_root).resolve()
    if (runtime_root/'locks').is_symlink():return
    for name in ('dashboard_generation_worker','unreal_capture'):
        folder=runtime_root/'locks'/name
        owner=folder/'owner.json'
        if folder.is_symlink() or owner.is_symlink() or not owner.exists():continue
        row=read_json(owner)
        info=process_info(row.get('pid'))
        if row.get('host')==socket.gethostname() and row.get('pid') in expected and (not info or info['state']=='Z'):
            owner.unlink();folder.rmdir()


def temporary_targets(root):
    targets=[]
    for sample in (root/'samples').iterdir() if (root/'samples').exists() else []:
        if sample.is_symlink() or not sample.is_dir():continue
        work=sample/'work'
        if work.exists() or work.is_symlink():targets.append(work)
    # Unfinished atomic writes/copies. Never traverse a symlinked directory.
    for directory,dirs,files in os.walk(root,followlinks=False):
        base=Path(directory)
        dirs[:]=[name for name in dirs if not (base/name).is_symlink() and base/name not in targets]
        targets.extend(base/name for name in files if name.endswith('.tmp'))
    return targets


def path_usage(path):
    count=0;size=0
    if path.is_symlink():return {'files':1,'bytes':path.lstat().st_size}
    if path.is_file():return {'files':1,'bytes':path.stat().st_size}
    for directory,dirs,files in os.walk(path,followlinks=False):
        base=Path(directory)
        for name in files:
            try:
                item=(base/name).lstat();count+=1;size+=item.st_size
            except FileNotFoundError:pass
        for name in list(dirs):
            item=base/name
            if item.is_symlink():count+=1;size+=item.lstat().st_size;dirs.remove(name)
    return {'files':count,'bytes':size}


def remove_path(path):
    if path.is_symlink() or path.is_file():path.unlink()
    elif path.exists():shutil.rmtree(path)


def reset_after_cleanup(root):
    for path in (root/'samples').glob('*/worker.json'):
        if path.parent.is_symlink() or path.is_symlink():continue
        old=read_json(path)
        completed=(path.parent/'COMPLETE.json').exists()
        state='complete' if completed else 'ready' if (path.parent/'CAPTURE_COMPLETE.json').exists() else 'awaiting_capture'
        atomic_json(path,{'state':state,'sample_id':path.parent.name,'attempts':0,
            'temporary_files_cleared':True,'previous_attempts':old.get('attempts',0)})


def profile_target(root):
    target=profile_directory(root)
    if target.parent.is_symlink():raise ValueError('Profiling root must not be a symlink')
    return target
