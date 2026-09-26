"""Detached single-connection transfer with per-file integrity checks."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time

ROOT=Path('/mnt/afs/202609/tyf-qwen-opd')
DEST=ROOT/'imports/20260927_ml2_base17_protocol'
SOURCE=Path('/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/migrations/20260927_acp_base17/migration_manifest.json')

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()

def write(path,data):
    p=path.with_suffix('.tmp');p.write_text(json.dumps(data,indent=2)+'\n');p.replace(path)

def serve():
    data=json.loads(SOURCE.read_text())
    with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz',compresslevel=1,dereference=True) as archive:
        for i,r in enumerate(data['files']):
            if not r['existing']:
                archive.add(r['source'],arcname=str(i),recursive=False)

def receive():
    manifest=DEST/'migration_manifest.json'
    data=json.loads(manifest.read_text()); records=data['files']
    started=time.monotonic();verified=set();total=0
    state=DEST/'transfer_state_v2.json'
    if (DEST/'transfer_acceptance.json').exists(): raise FileExistsError('Already accepted')
    for i,r in enumerate(records):
        p=Path(r['destination'])
        if not p.is_relative_to(ROOT): raise ValueError('Destination escapes AFS root')
        if r['existing']:
            if p.stat().st_size != r['size_bytes'] or sha(p) != r['sha256']:
                raise ValueError(f'Existing protected input differs: {p}')
            verified.add(i);total+=r['size_bytes']
    command=['ssh','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','HostKeyAlgorithms=ssh-ed25519',
             '-o',f'UserKnownHostsFile={DEST}/ml2_known_hosts','-o','ServerAliveInterval=30',
             '-o','ServerAliveCountMax=6','-p','36904','root@14.103.233.39',
             'python /tmp/stream_opd_base17_migration_20260927.py serve']
    process=subprocess.Popen(command,stdout=subprocess.PIPE,start_new_session=True)
    (DEST/'transfer_ssh.pid').write_text(str(process.pid)+'\n')
    try:
        with tarfile.open(fileobj=process.stdout,mode='r|gz') as archive:
            for member in archive:
                if not member.name.isdigit(): raise ValueError('Unexpected archive member')
                i=int(member.name)
                if i>=len(records) or i in verified: raise ValueError('Duplicate/unexpected member')
                r=records[i];p=Path(r['destination']);p.parent.mkdir(parents=True,exist_ok=True)
                if not member.isfile() or member.size!=r['size_bytes']:
                    raise ValueError(f'Wrong member type/size: {p}')
                temporary=p.with_name(p.name+'.stream_part')
                if temporary.exists() or p.exists(): raise FileExistsError(f'Never overwrite: {p}')
                write(state,dict(status='receiving',files_verified=len(verified),total_files=len(records),
                                 verified_bytes=total,file=str(p),source_connection_authenticated=True,
                                 elapsed_seconds=time.monotonic()-started))
                h=hashlib.sha256();size=0
                with archive.extractfile(member) as incoming, temporary.open('xb') as outgoing:
                    for b in iter(lambda:incoming.read(8*1024*1024),b''):
                        outgoing.write(b);h.update(b);size+=len(b)
                    outgoing.flush();os.fsync(outgoing.fileno())
                if size!=r['size_bytes'] or h.hexdigest()!=r['sha256']:
                    raise ValueError(f'Hash mismatch: {p}')
                temporary.replace(p);verified.add(i);total+=size
                if size>1_000_000_000 or len(verified)%100==0:
                    print(f'verified {len(verified)}/{len(records)} bytes={total} elapsed={time.monotonic()-started:.1f}',flush=True)
        process.stdout.close()
        code=process.wait(timeout=120)
        if code!=0 or verified!=set(range(len(records))):
            raise ValueError(f'Incomplete transfer: exit={code}, verified={len(verified)}')
        result=dict(passed=True,manifest_sha256=sha(manifest),files_verified=len(verified),total_bytes=total,
                    completed_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
        write(DEST/'transfer_acceptance.json',result)
        write(state,dict(status='complete',**result))
        print(json.dumps(result),flush=True)
    except BaseException as error:
        process.terminate()
        write(state,dict(status='failed',error=repr(error),files_verified=len(verified),verified_bytes=total))
        raise

if __name__=='__main__':
    if sys.argv[1]=='serve': serve()
    elif sys.argv[1]=='receive': receive()
    else: raise ValueError(sys.argv[1])
