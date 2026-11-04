// Render the real tier component with minimal browser globals; checks semantic output, not browser layout.
import assert from 'node:assert/strict';
import {writeFileSync} from 'node:fs';
import {pathToFileURL, fileURLToPath} from 'node:url';
const root=process.argv[2] || fileURLToPath(new URL('..',import.meta.url)), out=process.argv[3];
globalThis.document={addEventListener:()=>{},hidden:false};
globalThis.matchMedia=()=>({matches:false,addEventListener:()=>{}});
globalThis.localStorage={getItem:()=>null,setItem:()=>{}};
const {S}=await import(pathToFileURL(root+'/serve/web/state.js'));
const {MonitorPage}=await import(pathToFileURL(root+'/serve/web/monitor.js'));
function find(v,name) {
 if(Array.isArray(v))return v.flatMap(x=>find(x,name));
 if(!v||typeof v!=='object')return [];
 if(typeof v.type==='function'&&v.type.name===name)return [v];
 return find(v.props?.children,name);
}
const esc=s=>String(s).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('"','&quot;');
function render(v) {
 if(v==null||typeof v==='boolean')return '';
 if(Array.isArray(v))return v.map(render).join('');
 if(typeof v!=='object')return esc(v);
 if(typeof v.type==='function')return render(v.type(v.props));
 const attrs=Object.entries(v.props).filter(([k,val])=>k!=='children'&&!k.startsWith('on')&&val!=null&&val!==false).map(([k,val])=>` ${k}="${esc(val)}"`).join('');
 return `<${v.type}${attrs}>${render(v.props.children)}</${v.type}>`;
}
const now={time:Date.now()/1000,vram_used:8000,ram_used:16000,cpu_n:16000,vram_gb:42,ram_gb:110,vram_hit:.4,helper_held:1200,helper:32.5};
let outputs={};
for(const mode of ['helper','failed','plain']) {
 const n={...now};if(mode==='failed')n.helper_dead=1;
 if(mode==='plain'){delete n.helper_held;delete n.helper;}
 S.metrics={engine:{engine_kind:'ds4',cpu_experts:16000},tiers:{now:n}};
 const [tier]=find(MonitorPage(),'Tiers');assert(tier);
 const html=render(tier);outputs[mode]=html;
 assert(html.includes('flex-grow:8000') && html.includes('flex-grow:16000') && html.includes('flex-grow:0'));
 assert(!html.includes('flex-grow:9200')); // Helper copies must not be added to primary residency.

 assert(html.includes(mode==='plain'?'Where the experts live':'Primary cache and RAM backing'));
 if(mode==='plain')assert(!html.includes('Helper GPU:'));
 else {
  assert(html.includes('Primary VRAM'));
  assert(html.includes('CPU backing remains available for fallback'));
  assert(html.includes(mode==='failed'?'Helper unavailable':'1,200 profiled cached experts'));
  assert.equal(html.includes('the helper handles some'), mode==='helper');
 }
}
if(out)writeFileSync(out,JSON.stringify(outputs,null,2)+'\n');
console.log('PASS: actual Preact tier output, helper/failed/plain states; counts unchanged.');
