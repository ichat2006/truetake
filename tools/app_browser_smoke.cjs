// Uses the installed Windows Node/Chrome; no npm dependencies and no model calls.
const fs=require('node:fs'),path=require('node:path'),os=require('node:os');
const {spawn}=require('node:child_process');
const root=path.resolve(process.argv[2]);
const out=path.join(root,'outputs','app_screens');fs.mkdirSync(out,{recursive:true});
const profile=fs.mkdtempSync(path.join(os.tmpdir(),'truetake-app-'));
const delay=ms=>new Promise(r=>setTimeout(r,ms));
let chrome,ws,send;
(async()=>{
 chrome=spawn('C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',[
  '--headless=new','--disable-gpu','--no-first-run','--no-default-browser-check',
  '--enable-unsafe-swiftshader','--enable-unsafe-webgpu',
  '--autoplay-policy=no-user-gesture-required','--remote-debugging-port=0','--user-data-dir='+profile,'about:blank'],{stdio:'ignore'});
 const portFile=path.join(profile,'DevToolsActivePort');
 for(let i=0;!fs.existsSync(portFile)&&i<150;i++)await delay(100);
 const port=fs.readFileSync(portFile,'utf8').split('\n')[0];
 const pages=await(await fetch(`http://127.0.0.1:${port}/json/list`)).json();
 ws=new WebSocket(pages.find(t=>t.type==='page').webSocketDebuggerUrl);
 await new Promise((resolve,reject)=>{ws.onopen=resolve;ws.onerror=reject});
 let id=0;const pending=new Map();
 ws.onmessage=e=>{const m=JSON.parse(e.data),p=pending.get(m.id);if(p){pending.delete(m.id);m.error?p.reject(m.error):p.resolve(m.result)}};
 send=(method,params={})=>new Promise((resolve,reject)=>{const i=++id;pending.set(i,{resolve,reject});ws.send(JSON.stringify({id:i,method,params}))});
 const evaluate=async expression=>{const r=await send('Runtime.evaluate',{expression,awaitPromise:true,returnByValue:true});if(r.exceptionDetails)throw Error(JSON.stringify(r.exceptionDetails));return r.result.value};
 const wait=async(expression,timeout=30000)=>{const start=Date.now();while(Date.now()-start<timeout){if(await evaluate(`(()=>{try{return !!(${expression})}catch{return false}})()`))return;await delay(150)}throw Error('Timed out: '+expression)};
 const shot=async name=>{const r=await send('Page.captureScreenshot',{format:'png'});fs.writeFileSync(path.join(out,name),Buffer.from(r.data,'base64'));console.log('SCREENSHOT',name)};
 await send('Page.enable');await send('Emulation.setDeviceMetricsOverride',{width:1920,height:1080,deviceScaleFactor:1,mobile:false});
 await send('Page.navigate',{url:'http://localhost:8001/'});
 await wait("document.querySelector('#yaml-preview')?.textContent.includes('schema_version: 2')");
 const resume=process.argv.find(a=>a.startsWith('--run-id='));
 if(resume){await evaluate(`(async()=>{currentRun=await api('/api/runs/${resume.split('=')[1]}');results(currentRun)})()`)}else{
 await evaluate("document.querySelector('#load-example').click()");await delay(300);await shot('01_define.png');
 await evaluate("document.querySelector('#use-checklist').click()");await wait("document.body.dataset.step==='2'");
 await evaluate("document.querySelector('#load-demo').click()");await wait("document.querySelectorAll('.clip-card').length>=5");
 await wait("[...document.querySelectorAll('.thumbnail img')].every(i=>i.complete&&i.naturalWidth>0)");await shot('02_footage.png');
 await evaluate("document.querySelector('#start-review').click()");await wait("document.body.dataset.step==='3'");await delay(100);await shot('03_reviewing.png');
 await wait("document.body.dataset.step==='4'||document.querySelector('#message').classList.contains('error')",240000);
 if(await evaluate("document.body.dataset.step!=='4'"))throw Error(await evaluate("document.querySelector('#message').textContent"));
 }
 await wait("document.querySelector('#report').contentDocument?.querySelector('.start-here')");
 if(await evaluate("!!document.querySelector('#saved-percent')")){
  await wait("document.querySelector('#saved-percent').textContent.includes('%')");
  const dollars=await evaluate("document.querySelector('#saved-dollars').textContent");
  await evaluate("document.querySelector('#reviewer-rate').value=50;document.querySelector('#reviewer-rate').dispatchEvent(new Event('input'))");
  await wait(`document.querySelector('#saved-dollars').textContent!==${JSON.stringify(dollars)}`);
  await evaluate("document.querySelector('#reviewer-rate').value=25;document.querySelector('#reviewer-rate').dispatchEvent(new Event('input'))");
  await wait(`document.querySelector('#saved-dollars').textContent===${JSON.stringify(dollars)}`);
 }
 await shot('04_results.png');
 const thumbnails=await evaluate("(async()=>{const d=document.querySelector('#report').contentDocument;let nodes=[...d.querySelectorAll('svg.map image')];let results=await Promise.all(nodes.map(n=>new Promise(resolve=>{let i=new Image();i.onload=()=>resolve({ok:i.naturalWidth===320,width:i.naturalWidth});i.onerror=()=>resolve({ok:false});i.src=new URL(n.getAttribute('href'),d.baseURI).href})));return {count:nodes.length,results}})()");
 if(!thumbnails.count||!thumbnails.results.every(t=>t.ok))throw Error('Clip-map images failed: '+JSON.stringify(thumbnails));
 console.log('MAP THUMBNAILS',JSON.stringify(thumbnails));
 const cut=await evaluate("(()=>{const d=document.querySelector('#report').contentDocument;const c=d.querySelector('article[id$=\"_black_gap\"]');return c?{players:c.querySelectorAll('video').length,cut:c.textContent.includes('Proposed cut'),gap:!!c.querySelector('[title*=\"8–13\"]')}:null})()");
 if(cut&&(!cut.cut||cut.players!==2||!cut.gap))throw Error('Cut example missing side-by-side media or shaded gap: '+JSON.stringify(cut));
 const result=await evaluate("(async()=>{let run=await(await fetch('/api/runs/'+currentRun.id)).json();return {passed:run.status==='complete'&&run.runtime_api_calls===0&&run.detector_runs===0,run_id:run.id,counts:run.counts,runtime_api_calls:run.runtime_api_calls,detector_runs:run.detector_runs,report_url:run.report_url,start_tiles:document.querySelector('#report').contentDocument.querySelectorAll('.start-tile').length,clips:run.clips.map(c=>({clip_id:c.clip_id,verdict:c.verdict,license:c.license,cached:c.cached})),summary:document.querySelector('#result-title').textContent}})()");
 result.savings=await evaluate("currentRun.savings||null");
 fs.writeFileSync(path.join(out,'result.json'),JSON.stringify(result,null,2));console.log(JSON.stringify(result,null,2));
 if(!result.passed)process.exitCode=1;
 if(process.argv.includes('--examine')||process.env.TRUETAKE_EXAMINE==='1'){
  const examined=[];
  for(const [cid,file] of [['splice_P03_109_P04_104','05_examine_splice.png'],['P22_109_1851_1911_black_gap','06_examine_review.png']]){
   await evaluate(`document.querySelector('#report').contentDocument.querySelector('[data-examine="${cid}"]').click()`);
   await wait("document.querySelector('#examine-modal').open&&document.querySelector('#examine-viewer').src.includes('/viewer/0.42.0/')",120000);
   await wait("document.querySelector('#examine-viewer').contentDocument?.querySelector('canvas')",120000);
   await wait(`(()=>{const h=document.querySelector('#examine-viewer').contentWindow._handle;return h&&!h.has_panicked()&&new TextDecoder().decode(h.viewer_state()).includes('${cid}')})()`,120000);
   await delay(15000);await shot(file);
   const state=await evaluate("(()=>{const f=document.querySelector('#examine-viewer');return {src:f.src,canvas:!!f.contentDocument.querySelector('canvas'),state:new TextDecoder().decode(f.contentWindow._handle.viewer_state()).replace(/[^ -~]/g,' ').slice(0,1200)}})()");
   console.log('EXAMINE',cid,JSON.stringify(state));examined.push({clip_id:cid,...state});
   await evaluate("document.querySelector('#close-examine').click()");
  }
  fs.writeFileSync(path.join(out,'examine.json'),JSON.stringify(examined,null,2));
 }
})().catch(e=>{fs.writeFileSync(path.join(out,'result.json'),JSON.stringify({passed:false,error:String(e)},null,2));console.error(String(e));process.exitCode=1}).finally(async()=>{
 if(send)await send('Browser.close').catch(()=>{});if(ws)ws.close();if(chrome&&chrome.exitCode===null)chrome.kill();
 await delay(1000);try{fs.rmSync(profile,{recursive:true,force:true})}catch{}
});
