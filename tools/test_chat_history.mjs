// What the web chat sends as the history (serve/web/chats.js apiMessages): an answer's reasoning goes back as
// reasoning_content only when the server says the model's template keeps past thinking (/health history_thinking,
// Qwen3.x), split by tool round like the text; otherwise the request is as before.
//   node tools/test_chat_history.mjs
import assert from 'node:assert/strict';
import {pathToFileURL, fileURLToPath} from 'node:url';
const root=process.argv[2] || fileURLToPath(new URL('..',import.meta.url));
globalThis.document={addEventListener:()=>{},hidden:false,getElementById:()=>null};
globalThis.window=globalThis;
globalThis.matchMedia=()=>({matches:false,addEventListener:()=>{}});
globalThis.localStorage={getItem:()=>null,setItem:()=>{},removeItem:()=>{}};
globalThis.location={origin:'http://x',href:'http://x/',search:'',hash:''};
const {S}=await import(pathToFileURL(root+'/serve/web/state.js'));
const {apiMessages}=await import(pathToFileURL(root+'/serve/web/chats.js'));

const chat={system:'Be brief.',messages:[
 {role:'user',text:'What is 17*23?'},
 {role:'assistant',text:'391',reasoning:'17*20 + 17*3 = 391\n'},
 {role:'user',text:'Weather in Oslo?'},
 {role:'assistant',text:'Let me check.\n\nIt is 3 C.',reasoning:'Call the tool.\n\nGot 3.',
  tools:[{id:'c1',name:'w',round:0,at:13,rat:14,state:'done',arguments:{city:'Oslo'},result:'{"t":3}'}]},
 {role:'user',text:'Thanks'},
 {role:'assistant',text:'',reasoning:'half a thou',stopped:true},
 {role:'user',text:'Again?'},
]};

S.health={model:'m',history_thinking:false};
const off=apiMessages(chat);
assert.ok(off.every((m)=>!('reasoning_content' in m)),'no reasoning without history_thinking');
assert.deepEqual(off[2],{role:'assistant',content:'391'});

S.health={model:'m',history_thinking:true};
const on=apiMessages(chat);
assert.deepEqual(on[2],{role:'assistant',content:'391',reasoning_content:'17*20 + 17*3 = 391\n'});
const call=on[4], res=on[5], rest=on[6];
assert.equal(call.content,'Let me check.');
assert.equal(call.reasoning_content,'Call the tool.');
assert.equal(call.tool_calls[0].function.name,'w');
assert.equal(res.role,'tool');
assert.equal(rest.content,'It is 3 C.');
assert.equal(rest.reasoning_content,'\n\nGot 3.');
assert.deepEqual(on[8],{role:'assistant',content:''},'an answer stopped while thinking still goes back empty');
assert.equal(on.length,off.length);
console.log('chat history: OK');
