// The web chat's attachments (serve/web/chats.js): a file the browser cannot read goes to POST /attach and comes back
// as a text part or as page pictures; an unreadable one is a toast, never a silent drop; a video goes to the API as a
// video_url part, a picture as image_url.
//   node tools/test_chat_attach.mjs
import assert from 'node:assert/strict';
import {pathToFileURL, fileURLToPath} from 'node:url';
const root=process.argv[2] || fileURLToPath(new URL('..',import.meta.url));
globalThis.document={addEventListener:()=>{},hidden:false,getElementById:()=>null};
globalThis.window=globalThis;
globalThis.matchMedia=()=>({matches:false,addEventListener:()=>{}});
globalThis.localStorage={getItem:()=>null,setItem:()=>{},removeItem:()=>{}};
globalThis.location={origin:'http://x',href:'http://x/',search:'',hash:''};
const {S}=await import(pathToFileURL(root+'/serve/web/state.js'));
const {apiMessages,addFiles,addViaServer}=await import(pathToFileURL(root+'/serve/web/chats.js'));

// the API shape: pictures as image_url, videos as video_url, files as fenced text
const chat={system:'',messages:[{role:'user',text:'Look',files:[{name:'a.txt',text:'hello'}],
  images:[{kind:'image',name:'p.png',url:'data:image/png;base64,AA'},{kind:'video',name:'v.mp4',url:'data:video/mp4;base64,BB'}]}]};
S.health={model:'m'};
const [u]=apiMessages(chat);
assert.equal(u.content[0].text,'Look\n\nFile: a.txt\n```\nhello\n```');
assert.deepEqual(u.content[1],{type:'image_url',image_url:{url:'data:image/png;base64,AA'}});
assert.deepEqual(u.content[2],{type:'video_url',video_url:{url:'data:video/mp4;base64,BB'}});
const [light]=apiMessages(chat,chat.messages,true);
assert.equal(light.content[2].video_url.url,'data:,');

// the server's answers, as the page takes them
const calls=[];
let answer;
globalThis.fetch=async (url,opts)=>{calls.push({url,opts});return {json:async()=>answer};};
const file=(name,type='')=>new File([new Uint8Array([1,2,3])],name,{type});
S.attachments=[];S.current={messages:[]};S.toasts=[];
answer={kind:'text',format:'Word document',text:'Quarterly report',truncated:false};
await addViaServer(file('r é.docx'));
assert.equal(calls[0].url,'attach');
assert.equal(calls[0].opts.headers['Content-Type'],'application/octet-stream');
assert.equal(calls[0].opts.headers['X-File-Name'],'r%20%C3%A9.docx');
assert.deepEqual(S.attachments.at(-1),{kind:'file',name:'r é.docx',text:'Quarterly report'});

answer={kind:'images',format:'PDF',pages:2,images:[{data:'data:image/jpeg;base64,AA'},{data:'data:image/png;base64,BB'}]};
await addViaServer(file('scan.pdf','application/pdf'));
assert.deepEqual(S.attachments.slice(-2).map((a)=>[a.kind,a.name]),[['image','scan.pdf · page 1'],['image','scan.pdf · page 2']]);

const before=S.attachments.length;
answer={error:{message:'blob.bin is not a kind of file Maya reads'}};
await addViaServer(file('blob.bin'));
assert.equal(S.attachments.length,before,'nothing attached');
assert.ok(S.toasts.some((t)=>t.kind==='error' && t.text.includes('not a kind of file Maya reads')),'the reason is shown');

// routing: a PDF / Office file goes to the server; a video is refused with a reason when the model reads no videos
calls.length=0;
S.health={model:'m',images:true,videos:false};
addFiles([file('deck.pptx'),file('clip.mp4','video/mp4')]);
await new Promise((r)=>setTimeout(r,10));
assert.equal(calls.length,1,'the pptx went to the server, the video did not');
assert.ok(S.toasts.some((t)=>t.text.includes('reads pictures, not videos')),'the video refusal says why');
console.log('ok - chat attachments');
