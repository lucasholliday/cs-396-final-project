"""Loopback-only Swing Lab upload and processing service; standard-library HTTP.
Run locally, then open the printed URL. Files never leave this computer.
"""
import argparse,hmac,json,mimetypes,secrets,subprocess,sys,threading,uuid
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs,urlsplit,unquote
ROOT=Path(__file__).resolve().parents[1]
MAX_UPLOAD=2*1024**3

class App(ThreadingHTTPServer):
    def __init__(self,address):
        super().__init__(address,Handler);self.token=secrets.token_urlsafe(32);self.jobs={};self.busy=threading.Lock()
    def process(self,jid,path,start):
        job=self.jobs[jid]
        try:
            result=subprocess.run([sys.executable,str(ROOT/'scripts/analyze_swing.py'),'--video',str(path),'--start',str(start),'--name',jid],capture_output=True,text=True,timeout=600)
            (path.parent/'processing.log').write_text(result.stdout+'\n'+result.stderr)
            if result.returncode:job.update(status='failed',message='Processing failed. Check the local processing.log for details; confirm the file contains a complete 3.5-second swing window.')
            else:job.update(status='complete',url=f'/outputs/new_swings/{jid}/index.html')
        except Exception as e:job.update(status='failed',message=str(e))
        finally:self.busy.release()

class Handler(BaseHTTPRequestHandler):
    def send_bytes(self,code,data,kind='application/json'):
        self.send_response(code);self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(data)));self.send_header('X-Content-Type-Options','nosniff');self.end_headers();self.wfile.write(data)
    def json(self,code,data):self.send_bytes(code,json.dumps(data).encode())
    def authorized(self,query):return hmac.compare_digest(query.get('token',[''])[0],self.server.token)
    def local_host(self):return self.headers.get('Host')==f'127.0.0.1:{self.server.server_port}'
    def do_GET(self):
        if not self.local_host():return self.json(403,{'error':'Use the printed loopback URL'})
        u=urlsplit(self.path);q=parse_qs(u.query)
        if u.path=='/':
            if not self.authorized(q):return self.json(403,{'error':'Open the complete URL printed by the launcher'})
            return self.send_bytes(200,(ROOT/'scripts/upload_template.html').read_bytes(),'text/html; charset=utf-8')
        if u.path.startswith('/api/status/'):
            if not self.authorized(q):return self.json(403,{'error':'Invalid session token'})
            return self.json(200,self.server.jobs.get(u.path.rsplit('/',1)[-1],{'status':'unknown'}))
        relative=unquote(u.path).lstrip('/');path=(ROOT/relative).resolve()
        allowed=[(ROOT/'outputs').resolve(),(ROOT/'reports').resolve()]
        if not any(path.is_relative_to(base) for base in allowed) or not path.is_file():return self.json(404,{'error':'Not found'})
        # Stream media ranges so native video controls can seek without loading full clips.
        size=path.stat().st_size;start,end=0,size-1;range_header=self.headers.get('Range');code=200
        if range_header:
            import re
            match=re.fullmatch(r'bytes=(\d+)-(\d*)',range_header)
            if not match:return self.json(416,{'error':'Unsupported range'})
            start=int(match[1]);end=min(size-1,int(match[2])) if match[2] else size-1;code=206
            if start>end:return self.json(416,{'error':'Range outside file'})
        self.send_response(code);self.send_header('Content-Type',mimetypes.guess_type(path.name)[0] or 'application/octet-stream');self.send_header('Content-Length',str(end-start+1));self.send_header('Accept-Ranges','bytes');
        if path.suffix=='.html':self.send_header('Cache-Control','no-store')
        self.send_header('X-Content-Type-Options','nosniff')
        if code==206:self.send_header('Content-Range',f'bytes {start}-{end}/{size}')
        self.end_headers()
        with path.open('rb') as f:
            f.seek(start);remaining=end-start+1
            while remaining:
                block=f.read(min(1024*1024,remaining))
                if not block:break
                self.wfile.write(block);remaining-=len(block)
    def do_POST(self):
        if not self.local_host():return self.json(403,{'error':'Loopback host required'})
        u=urlsplit(self.path);q=parse_qs(u.query);origin=self.headers.get('Origin')
        if not self.authorized(q) or (origin and origin!=f'http://127.0.0.1:{self.server.server_port}'):return self.json(403,{'error':'Invalid session or origin'})
        if u.path!='/api/analyze':return self.json(404,{'error':'Not found'})
        try:
            length=int(self.headers.get('Content-Length','0'));start=float(q.get('start',['0'])[0]);suffix=q.get('extension',[''])[0].lower()
            if not 0<length<=MAX_UPLOAD or not 0<=start<86400 or suffix not in ['.mp4','.mov']:raise ValueError()
        except ValueError:return self.json(400,{'error':'Choose MOV/MP4 up to 2 GiB and a valid clip start time'})
        if not self.server.busy.acquire(blocking=False):return self.json(409,{'error':'A video is already processing; wait for it to finish'})
        jid='swing_'+uuid.uuid4().hex[:12];folder=ROOT/'outputs/new_swings'/jid;folder.mkdir(parents=True);path=folder/('source'+suffix)
        dispatched=False
        try:
            self.connection.settimeout(180)
            with path.open('xb') as f:
                remaining=length
                while remaining:
                    b=self.rfile.read(min(1024*1024,remaining))
                    if not b:raise ValueError('Upload ended early')
                    f.write(b);remaining-=len(b)
            self.server.jobs[jid]={'status':'processing','id':jid};threading.Thread(target=self.server.process,args=(jid,path,start),daemon=True).start();dispatched=True;self.json(202,{'id':jid,'status':'processing'})
        except Exception as e:
            if not dispatched:self.server.busy.release()
            self.json(400,{'error':str(e)})
    def log_message(self,format,*args):pass # Avoid printing token-bearing URLs in request logs.

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,default=8765);args=parser.parse_args();server=App(('127.0.0.1',args.port));print(f'Swing Lab: http://127.0.0.1:{server.server_port}/?token={server.token}',flush=True);print('Local processing only. Keep this terminal open; Ctrl+C stops the server.',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:server.server_close()
