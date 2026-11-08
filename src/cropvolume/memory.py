"""Approximate stage memory costs without loading meshes, images, torch or a GPU."""
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time

from .io import child, read_json

GIB = 1024**3
MODEL = 'stage-memory-local-overlap-v2'


def linux_memory(proc=Path('/proc'), cgroup=Path('/sys/fs/cgroup')):
    try:
        values = {line.split(':', 1)[0]: int(line.split()[1])*1024
                  for line in (proc/'meminfo').read_text().splitlines() if ':' in line}
        total, available = values['MemTotal'], values['MemAvailable']
    except (OSError, ValueError, KeyError, IndexError):
        return {'total_bytes': None, 'available_bytes': None, 'swap_bytes': None}
    # Respect container limits as well as the WSL VM limit exposed by meminfo.
    try:
        membership = next(line.split(':', 2)[2] for line in (proc/'self/cgroup').read_text().splitlines() if line.startswith('0::'))
        path = cgroup / membership.lstrip('/')
        while path.is_relative_to(cgroup):
            try:
                limit = int((path/'memory.max').read_text())
                used = int((path/'memory.current').read_text())
                total, available = min(total, limit), min(available, max(0, limit-used))
            except (OSError, ValueError):
                pass
            if path == cgroup:
                break
            path = path.parent
    except (OSError, StopIteration):
        pass
    return {'total_bytes': total, 'available_bytes': min(total, available), 'swap_bytes': values.get('SwapTotal', 0)}


WINDOWS_PROBE = r'''
$ErrorActionPreference='Stop'
$os=Get-CimInstance Win32_OperatingSystem
$devices=Get-ChildItem 'HKLM:\SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}' -ErrorAction SilentlyContinue | Where-Object {$_.PSChildName -match '^\d{4}$'} | ForEach-Object {Get-ItemProperty $_.PSPath}
$gpu=$devices | Where-Object {[int64]$_."HardwareInformation.qwMemorySize" -gt 0} | Sort-Object {[int64]$_."HardwareInformation.qwMemorySize"} -Descending | Select-Object -First 1
[pscustomobject]@{host_ram=@{total_bytes=([int64]$os.TotalVisibleMemorySize*1024);available_bytes=([int64]$os.FreePhysicalMemory*1024)};gpu=@{name=$gpu.DriverDesc;total_bytes=$gpu."HardwareInformation.qwMemorySize";available_bytes=$null;source='Windows 64-bit adapter registry capacity; free VRAM unavailable'}} | ConvertTo-Json -Depth 5 -Compress
'''


class HardwareMemory:
    def __init__(self):
        self.cached = {}
        self.updated = 0
        self.pending = False
        self.lock = threading.Lock()

    def _probe(self):
        try:
            powershell = shutil.which('powershell.exe')
            if powershell:
                result = subprocess.run([powershell, '-NoProfile', '-NonInteractive', '-Command', WINDOWS_PROBE],
                    capture_output=True, text=True, timeout=12, check=True)
                value = json.loads(result.stdout)
            else:
                value = {'host_ram': None, 'gpu': None}
        except (OSError, ValueError, subprocess.SubprocessError):
            value = {'host_ram': None, 'gpu': None}
        with self.lock:
            self.cached = value
            self.updated = time.time()
            self.pending = False

    def snapshot(self):
        with self.lock:
            if not self.pending and time.time()-self.updated > 30:
                self.pending = True
                threading.Thread(target=self._probe, daemon=True).start()
            value = {**self.cached, 'probe_pending': self.pending, 'host_sampled_unix': self.updated or None}
        return {**value, 'worker_ram': linux_memory(), 'sampled_unix': time.time()}


def ply_counts(path):
    counts = {}
    with path.open('rb') as stream:
        if stream.readline().strip() != b'ply':
            raise ValueError('Memory estimate requires an approved PLY mesh')
        for _ in range(256):
            line = stream.readline(4096).decode('ascii').strip()
            if line.startswith('element '):
                _, name, count = line.split()
                counts[name] = int(count)
            if line == 'end_header':
                if counts.get('vertex', 0) <= 0 or counts.get('face', 0) <= 0:
                    raise ValueError('Mesh header has no vertices/faces')
                return {'vertices': counts['vertex'], 'faces': counts['face']}
    raise ValueError('Mesh header is incomplete')


def byte_range(low, high):
    return {'low_bytes': math.ceil(low), 'high_bytes': math.ceil(max(low, high))}


def estimate_memory(settings, meshes, hardware=None, counts=None):
    from .generation import validate_settings, plant_count_range
    s = validate_settings(settings, derive_population=counts is None)
    counts = counts or plant_count_range(s['field_width_m'], s['field_length_m'], s['layout'], s['scale_max'], s['spacing_m'])
    selected = [m for m in meshes if s['species'] == 'mixed' or m['species'] == s['species']]
    if not selected:
        raise ValueError('No mesh complexity data for selected species')
    pixels = s['image_width']*s['image_height']
    gaussians = s['max_gaussians']
    faces = max(m['faces'] for m in selected)
    vertices = max(m['vertices'] for m in selected)
    unique_faces = sum(m['faces'] for m in selected)
    # Cached sums only need one metadata row and transformed AABB per plant.
    # An overlap group holds mesh copies plus a variable Boolean workspace.
    overlap_group = min(counts[1], 64)
    overlap_bytes = (vertices*24 + faces*24)*4 + faces*512
    volume = byte_range(.5*GIB+counts[0]*4096, GIB+counts[1]*8192+overlap_group*overlap_bytes)
    volume_worst = GIB+counts[1]*8192+counts[1]*overlap_bytes
    unreal_ram = byte_range(4*GIB+unique_faces*96+counts[0]*16384+pixels*24,
                            8*GIB+unique_faces*256+counts[1]*65536+pixels*64)
    unreal_gpu = byte_range(.75*GIB+unique_faces*64+pixels*48,
                            2*GIB+unique_faces*128+pixels*160)
    training_views = math.ceil(s['views']*.75)
    descriptors = training_views*4096*128
    preparation = byte_range(.5*GIB+descriptors*2+pixels*16,
                            1.5*GIB+descriptors*8+pixels*64)
    training_ram = byte_range(.75*GIB+min(GIB,s['views']*pixels*3)+gaussians*256,
                             2*GIB+min(GIB,s['views']*pixels*3)+gaussians*1024+pixels*48)
    # 59 float32 parameters, gradients and two Adam moments ≈944 B/Gaussian;
    # allow extra space for growth, rasterization, autograd and context/caches.
    training_gpu = byte_range(.5*GIB+gaussians*1100+pixels*64,
                             1.5*GIB+gaussians*3000+pixels*256)
    stages = [
        {'stage':'volume','label':'Cached plant volumes + overlap groups','ram':volume,'gpu':byte_range(0,0),'memory_domain':'worker'},
        {'stage':'capture','label':'Unreal capture','ram':unreal_ram,'gpu':unreal_gpu,'memory_domain':'host'},
        {'stage':'preparation','label':'RGB preparation / matching','ram':preparation,'gpu':byte_range(0,0),'memory_domain':'worker'},
        {'stage':'training','label':'Gaussian training / export','ram':training_ram,'gpu':training_gpu,'memory_domain':'worker'}]
    worker = byte_range(max(x['ram']['low_bytes'] for x in stages if x['memory_domain']=='worker'),
                        max(x['ram']['high_bytes'] for x in stages if x['memory_domain']=='worker'))
    gpu = byte_range(max(x['gpu']['low_bytes'] for x in stages), max(x['gpu']['high_bytes'] for x in stages))
    hardware = hardware or {}
    warnings = []
    for label, demand, supply in [('WSL/worker RAM', worker, hardware.get('worker_ram')),
                                  ('Unreal host RAM', unreal_ram, hardware.get('host_ram'))]:
        available = (supply or {}).get('available_bytes')
        if available is not None and demand['high_bytes'] > available:
            warnings.append({'kind':'memory','resource':label,'message':f'{label}: estimated upper range exceeds currently available RAM.'})
        elif available is not None and demand['high_bytes'] > available*.8:
            warnings.append({'kind':'headroom','resource':label,'message':f'{label}: limited memory headroom with these settings.'})
    worker_available = (hardware.get('worker_ram') or {}).get('available_bytes')
    if worker_available is not None and volume_worst > worker_available and counts[1] > overlap_group:
        warnings.append({'kind':'conditional','resource':'Overlap geometry',
            'message':'Dense local overlap calculations could exceed available worker RAM. See the overlap assumptions and worst case below.'})
    gpu_capacity = (hardware.get('gpu') or {}).get('total_bytes')
    if gpu_capacity and gpu['high_bytes'] > gpu_capacity*.8:
        warnings.append({'kind':'memory','resource':'GPU VRAM','message':'Estimated GPU memory approaches or exceeds adapter capacity; free VRAM is unknown.'})
    return {'schema':'cropvolume-memory-estimate-v1','model':MODEL,'plant_count':counts,
        'worker_ram':worker,'unreal_ram':unreal_ram,'gpu_memory':gpu,'stages':stages,
        'hardware':hardware,'warnings':warnings,'sequential_pairs':True,
        'volume_overlap_worst_case_bytes':math.ceil(volume_worst),
        'assumptions':{'overlap_group_plants':overlap_group,'image_cache_limit_bytes':GIB,
            'max_features_per_view':4096,'parallel_pairs':1},
        'notes':['Approximate working ranges, not measured peaks or guaranteed limits.',
            'Pairs run sequentially; pair count increases runtime/storage, not peak memory.',
            'Cached volumes are summed directly; large connected groups are processed through local overlap neighborhoods.',
            'Dense local overlaps and projected Gaussians can exceed these ranges. Swap is not counted as RAM.',
            'Windows host RAM and WSL RAM share physical memory; available capacity changes as other applications run.']}


class MemoryEstimator:
    def __init__(self, catalog):
        self.catalog = Path(catalog)
        self.profiles = None
        self.stamp = None
        self.hardware = HardwareMemory()
        self.lock = threading.Lock()

    def estimate(self, settings, counts=None):
        with self.lock:
            stamp = self.catalog.stat().st_mtime_ns
            if self.profiles is None or stamp != self.stamp:
                catalog = read_json(self.catalog)
                self.profiles = [{'species':a['species'], **ply_counts(child(self.catalog.parent,a['mesh']))}
                    for a in catalog['assets'] if a['approved']]
                self.stamp = stamp
            meshes = list(self.profiles)
        return estimate_memory(settings, meshes, self.hardware.snapshot(), counts)
