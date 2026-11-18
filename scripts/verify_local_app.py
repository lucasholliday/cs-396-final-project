"""HTTP contract and optional real processing smoke test; not browser UI testing."""
import argparse,json,threading,time
from pathlib import Path
from urllib.request import Request,urlopen
from urllib.error import HTTPError
from local_app import App,ROOT

def run(process=False):
 server=App(('127.0.0.1',0));threading.Thread(target=server.serve_forever,daemon=True).start();base=f'http://127.0.0.1:{server.server_port}';token=server.token;checks=[]
 def request(path,body=None,headers=None):
  req=Request(base+path,data=body,headers=headers or {})
  try:
   with urlopen(req,timeout=180) as r:return r.status,r.read(),r.headers
  except HTTPError as e:return e.code,e.read(),e.headers
 try:
  assert request('/')[0]==403;checks.append('landing page requires session token')
  status,body,_=request('/?token='+token);assert status==200 and b'Analyze your video' in body;checks.append('upload page delivered')
  assert request('/data/manifests/shot_manifest.csv')[0]==404;assert request('/outputs/../data/manifests/shot_manifest.csv')[0]==404;checks.append('static serving restricted to outputs/reports')
  path='/outputs/pose/sid_1_01_input.mp4';status,body,headers=request(path,headers={'Range':'bytes=0-99'});assert status==206 and len(body)==100 and headers['Content-Range'].startswith('bytes 0-99/');checks.append('video byte-range seeking')
  assert request('/api/analyze?token=wrong&start=0&extension=.mp4',b'x')[0]==403;assert request('/api/analyze?token='+token+'&start=0&extension=.mp4',b'x',{'Origin':'https://example.invalid'})[0]==403;checks.append('invalid session and cross-origin processing rejected')
  assert request('/api/analyze?token='+token+'&start=-1&extension=.mp4',b'x')[0]==400;checks.append('invalid clip request rejected')
  result=None
  if process:
   clip=(ROOT/'outputs/pose/sid_1_01_input.mp4').read_bytes();status,body,_=request('/api/analyze?token='+token+'&start=0&extension=.mp4',clip,{'Content-Type':'application/octet-stream'});assert status==202,body;jid=json.loads(body)['id'];deadline=time.monotonic()+180
   while time.monotonic()<deadline:
    _,body,_=request('/api/status/'+jid+'?token='+token);job=json.loads(body)
    if job['status'] in ['complete','failed']:break
    time.sleep(1)
   assert job['status']=='complete',job;status,html,_=request(job['url']);assert status==200 and b'Experimental video-only carry estimate' in html;checks.append('real clip upload, pose processing, reference prediction and result delivery');result=job['url']
  report=dict(checks=checks,result=result,browser_ui_test='not performed; browser host unavailable',accuracy_validation='none; smoke input is a reused training-session clip')
  (ROOT/'reports/local_app_verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
 finally:server.shutdown();server.server_close()
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--process',action='store_true');args=parser.parse_args();run(args.process)
