"""Build a portable source release with assets, hashes, and no machine/run data."""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile
import tomllib

ROOT=Path(__file__).resolve().parents[1]
INCLUDE=('src','tools','tests','configs','dashboard','simulator','assets','third_party','pipeline','docs',
         'README.md','run.py','pyproject.toml','.gitignore','.gitattributes','THIRD_PARTY.md','cropvolume.example.json')
SKIP={'__pycache__','.pytest_cache','.git','Binaries','Intermediate','Saved','DerivedDataCache','*.egg-info'}
SUFFIXES=('.pyc','.pyo','.o','.so','.pyd','.dll','.tar.gz','.zip','.tmp')


def included_files(root):
    for name in INCLUDE:
        path=root/name
        if not path.exists():continue
        for item in sorted(path.rglob('*')) if path.is_dir() else [path]:
            relative=item.relative_to(root)
            if any(p in SKIP or p.endswith('.egg-info') for p in relative.parts) or item.name.endswith(SUFFIXES):continue
            if item.is_symlink():raise ValueError('Release cannot contain symlinks: '+str(relative))
            if item.is_file():yield item


def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def build_release(root,output):
    root,output=Path(root).resolve(),Path(output).resolve()
    version=tomllib.loads((root/'pyproject.toml').read_text())['project']['version']
    prefix='cropvolume-'+version
    files=list(included_files(root))
    manifest={'schema':'cropvolume-release-v1','version':version,
        'files':{p.relative_to(root).as_posix():digest(p) for p in files}}
    output.parent.mkdir(parents=True,exist_ok=True)
    temporary=output.with_name(output.name+'.tmp')
    def info(name,size=0):
        value=tarfile.TarInfo(prefix+'/'+name);value.size=size;value.mode=0o644;value.mtime=0
        return value
    try:
        with temporary.open('wb') as raw,gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0,compresslevel=6) as compressed,tarfile.open(fileobj=compressed,mode='w') as archive:
            for path in files:
                with path.open('rb') as stream:archive.addfile(info(path.relative_to(root).as_posix(),path.stat().st_size),stream)
            payload=(json.dumps(manifest,indent=2,sort_keys=True)+'\n').encode()
            archive.addfile(info('release-manifest.json',len(payload)),io.BytesIO(payload))
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    checksum=digest(output)
    output.with_name(output.name+'.sha256').write_text(checksum+'  '+output.name+'\n')
    return {'archive':str(output),'sha256':checksum,'files':len(files),'bytes':output.stat().st_size}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    version=tomllib.loads((ROOT/'pyproject.toml').read_text())['project']['version']
    print(json.dumps(build_release(ROOT,args.output or ROOT/'dist'/f'cropvolume-{version}.tar.gz'),indent=2))


if __name__=='__main__':main()
