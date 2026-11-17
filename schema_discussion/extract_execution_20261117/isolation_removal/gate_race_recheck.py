"""Real ps/proc enumeration race: fence first, then converge under normal observation.

Adapted from the independent review reproducer; no process-table rows are falsified.
"""
import ctypes,json,os,signal,subprocess,sys,tempfile,time
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from schema_extract.tools import processes
from schema_extract.tools.common import digest,write_json
# Collect the deliberately escaped test descendant ourselves after the repro.
assert ctypes.CDLL(None).prctl(36,1,0,0,0)==0
root=Path(tempfile.mkdtemp(prefix='extract-gate-race-review-'))
child=root/'child.py'
child.write_text('''import os,sys,time\nfrom pathlib import Path\nr=Path(sys.argv[1])\nos.setsid()\nwhile True:\n p=os.getpid(); t=r/('pid.'+str(p)); t.write_text('ready')\n temporary=r/('current.'+str(p)); temporary.write_text(str(p)); os.replace(temporary,r/'current')\n while not (r/('next.'+str(p))).exists(): time.sleep(.001)\n if os.fork(): os._exit(0)\n os.setsid()\n''')
attempt={'attempt_id':str(uuid4()),'directory':str(root)}
launch={'protocol':processes.GATE_PROTOCOL,'launch_id':str(uuid4()),'attempt_id':attempt['attempt_id'],'attempt_sha256':digest(attempt),'status':'released'}
write_json(root/'launch.json',launch)
r,w=os.pipe()
gate=subprocess.Popen([sys.executable,'-B',processes.__file__,'--launch-gate',str(r),str(root/'launch.json'),launch['launch_id'],attempt['attempt_id'],digest(attempt),str(root),sys.executable,str(child),str(root)],pass_fds=(r,),start_new_session=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
os.close(r);os.write(w,b'1');os.close(w)
deadline=time.monotonic()+5
while not (root/'current').exists() and time.monotonic()<deadline:time.sleep(.005)
identity=processes.process_table()[gate.pid]
original=processes._linux_process
handoffs=[]
def at_stat(pid,boot_id):
 # Do not falsify ps or /proc rows. Schedule the real child fork+exit after
 # ps has enumerated the parent and before /proc/<parent>/stat is read.
 current=int((root/'current').read_text())
 if pid==current and gate.poll() is None:
  (root/('next.'+str(pid))).touch()
  until=time.monotonic()+2
  while time.monotonic()<until:
   nextpid=int((root/'current').read_text())
   if nextpid!=pid and not Path('/proc/'+str(pid)).exists():break
   time.sleep(.001)
  handoffs.append([pid,nextpid])
 return original(pid,boot_id)
try:
 with patch.object(processes,'_linux_process',side_effect=at_stat):
  result=processes.terminate_owned(identity,grace=1,cwd=root)
 escaped=int((root/'current').read_text())
 alive=processes.process_table().get(escaped)
 first={'root':str(root),'termination':result,'handoffs':handoffs,'escaped_pid':escaped,
        'escaped_alive':bool(alive and not alive['stat'].startswith('Z')),
        'gate_alive':gate.poll() is None,'gate_reaped':processes.gate_reaped(attempt)}
 assert result['confirmed'] is False and first['gate_alive'] and first['escaped_alive']
 assert not first['gate_reaped'] and all(action['pid']!=gate.pid for action in result['actions'])
 # With the forced enumeration race removed, ordinary cancellation must converge.
 settled=processes.terminate_owned(identity,grace=1,cwd=root)
 gate.wait(timeout=3)
 current=processes.process_table().get(escaped)
 second={'termination':settled,'gate_reaped':processes.gate_reaped(attempt),
         'descendant_alive':bool(current and not current['stat'].startswith('Z'))}
 assert settled['confirmed'] is True and second['gate_reaped'] and not second['descendant_alive']
 print(json.dumps({'race':first,'settled':second},indent=2))

finally:
 for p in root.glob('pid.*'):
  try:os.kill(int(p.name.split('.')[1]),signal.SIGKILL)
  except ProcessLookupError:pass
 if gate.poll() is None:gate.kill()
 gate.wait()
 end=time.monotonic()+3
 while time.monotonic()<end:
  try:
   pid,_=os.waitpid(-1,os.WNOHANG)
   if not pid:time.sleep(.01)
  except ChildProcessError:break
