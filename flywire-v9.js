const $ = s => document.querySelector(s);
const ATLAS_URL = 'https://cdn.jsdelivr.net/gh/lixiang1076/fly-brain@main/data/mushroom_body_neurons.json';
const CODEX_URL = 'https://codex.flywire.ai/api/download_resource?data_product=connections_princeton&dataset=fafb';
const DB_NAME = 'flytalk-flywire-v9';
const DB_VER = 1;
const GRAPH_KEY = 'mb-v783';
const WEIGHT_KEY = 'mb-v783-gains';

let atlas = null;
let graph = null;
let rt = null;
let lastTrace = null;
let dbPromise = null;
let saveTimer = null;

function fmt(n){ return Number(n || 0).toLocaleString('pt-BR'); }
function pct(x){ return Number.isFinite(x) ? (x * 100).toFixed(1) + '%' : '—'; }
function set(id, text){ const el = $(id); if (el) el.textContent = text; }
function status(text){ set('#event', text); }
function msg(kind, text){
  const box = $('#chat');
  const el = document.createElement('div');
  el.className = 'msg ' + kind;
  if (kind !== 'sys') {
    const who = document.createElement('div');
    who.className = 'who';
    who.textContent = kind === 'user' ? 'ESTÍMULO' : 'SAÍDA MBON';
    el.appendChild(who);
  }
  el.appendChild(document.createTextNode(text));
  box.appendChild(el);
  while (box.children.length > 60) box.firstChild.remove();
  box.scrollTop = box.scrollHeight;
}

function section(text, from, to){
  const a = text.indexOf(from);
  if (a < 0) return '';
  const b = to ? text.indexOf(to, a + from.length) : text.length;
  return text.slice(a, b < 0 ? text.length : b);
}
function idsFrom(raw){ return [...new Set(raw.match(/720575940\d+/g) || [])]; }

async function loadAtlas(){
  set('#atlasState', 'CARREGANDO');
  const res = await fetch(ATLAS_URL, {cache:'force-cache'});
  if (!res.ok) throw new Error('falha ao carregar atlas: HTTP ' + res.status);
  const raw = await res.text();
  const kc = idsFrom(section(raw, '"kenyon_cells"', '"mbon"'));
  const mbon = idsFrom(section(raw, '"mbon"', '"dan_pam_reward"'));
  const pam = idsFrom(section(raw, '"dan_pam_reward"', '"dan_ppl_punishment"'));
  const ppl = idsFrom(section(raw, '"dan_ppl_punishment"', null));
  if (kc.length < 1000 || mbon.length < 20) throw new Error('atlas FlyWire incompleto');
  const role = new Map();
  kc.forEach(x => role.set(x, 'KC'));
  mbon.forEach(x => role.set(x, 'MBON'));
  pam.forEach(x => role.set(x, 'PAM'));
  ppl.forEach(x => role.set(x, 'PPL'));
  atlas = {kc, mbon, pam, ppl, role, all:new Set([...kc,...mbon,...pam,...ppl])};
  set('#atlasState', 'READY');
  set('#kcCount', fmt(kc.length));
  set('#mbonCount', fmt(mbon.length));
  set('#pamCount', fmt(pam.length));
  set('#pplCount', fmt(ppl.length));
  set('#neuronCount', fmt(role.size));
  status(`atlas v783 carregado · ${fmt(role.size)} neurônios do mushroom body catalogados`);
}

function openDB(){
  if (dbPromise) return dbPromise;
  dbPromise = new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VER);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains('store')) db.createObjectStore('store');
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
  return dbPromise;
}
async function dbGet(key){
  const db = await openDB();
  return new Promise((resolve,reject)=>{
    const tx = db.transaction('store','readonly');
    const r = tx.objectStore('store').get(key);
    r.onsuccess = () => resolve(r.result);
    r.onerror = () => reject(r.error);
  });
}
async function dbPut(key, value){
  const db = await openDB();
  return new Promise((resolve,reject)=>{
    const tx = db.transaction('store','readwrite');
    tx.objectStore('store').put(value,key);
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}
async function dbDelete(key){
  const db = await openDB();
  return new Promise((resolve,reject)=>{
    const tx = db.transaction('store','readwrite');
    tx.objectStore('store').delete(key);
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}

function parseCSVLine(line){
  if (!line.includes('"')) return line.split(',');
  const out=[]; let cur='', q=false;
  for(let i=0;i<line.length;i++){
    const c=line[i];
    if(c==='"'){
      if(q && line[i+1]==='"'){cur+='"';i++;} else q=!q;
    } else if(c===',' && !q){out.push(cur);cur='';} else cur+=c;
  }
  out.push(cur); return out;
}
function findCol(headers, names){
  for(const n of names){ const i=headers.indexOf(n); if(i>=0) return i; }
  return -1;
}

async function processConnectionStream(stream, compressed=false, source='arquivo'){
  if (!atlas) throw new Error('atlas ainda não carregado');
  if (compressed){
    if (!('DecompressionStream' in window)) throw new Error('este navegador não suporta gzip streaming');
    stream = stream.pipeThrough(new DecompressionStream('gzip'));
  }
  const reader = stream.pipeThrough(new TextDecoderStream()).getReader();
  let carry='', header=null, preI=-1, postI=-1, synI=-1;
  let rows=0, keptRows=0, keptSyn=0, lastUI=performance.now();
  const agg = new Map();
  set('#graphState','IMPORTANDO');
  set('#importProgress','iniciando leitura…');
  status('filtrando conexões oficiais no navegador…');

  while(true){
    const {value,done}=await reader.read();
    if(done) break;
    carry += value;
    let start=0, nl;
    while((nl=carry.indexOf('\n',start))>=0){
      let line=carry.slice(start,nl); start=nl+1;
      if(line.endsWith('\r')) line=line.slice(0,-1);
      if(!line) continue;
      if(!header){
        header=parseCSVLine(line).map(x=>x.trim().replace(/^"|"$/g,'').toLowerCase());
        preI=findCol(header,['pre_root_id','pre_pt_root_id']);
        postI=findCol(header,['post_root_id','post_pt_root_id']);
        synI=findCol(header,['syn_count','synapse_count','n_synapses','count','weight']);
        if(preI<0||postI<0||synI<0) throw new Error('cabeçalho inesperado: '+header.slice(0,12).join(', '));
        continue;
      }
      rows++;
      const parts=line.split(',');
      const pre=(parts[preI]||'').replace(/"/g,'').trim();
      const post=(parts[postI]||'').replace(/"/g,'').trim();
      if(!atlas.all.has(pre) || !atlas.all.has(post)) continue;
      const n=Number((parts[synI]||'0').replace(/"/g,''))||0;
      if(n<=0) continue;
      const key=pre+'|'+post;
      agg.set(key,(agg.get(key)||0)+n);
      keptRows++; keptSyn+=n;
      const now=performance.now();
      if(now-lastUI>250){
        set('#importProgress',`${fmt(rows)} linhas lidas · ${fmt(agg.size)} pares MB · ${fmt(keptSyn)} sinapses`);
        lastUI=now;
        await new Promise(requestAnimationFrame);
      }
    }
    carry=carry.slice(start);
  }
  if(carry.trim() && header){
    const parts=carry.trim().split(',');
    rows++;
    const pre=(parts[preI]||'').replace(/"/g,'').trim();
    const post=(parts[postI]||'').replace(/"/g,'').trim();
    if(atlas.all.has(pre)&&atlas.all.has(post)){
      const n=Number((parts[synI]||'0').replace(/"/g,''))||0;
      if(n>0){const key=pre+'|'+post;agg.set(key,(agg.get(key)||0)+n);keptRows++;keptSyn+=n;}
    }
  }
  const edges=[];
  let plastic=0;
  for(const [key,count] of agg){
    const p=key.indexOf('|'), pre=key.slice(0,p), post=key.slice(p+1);
    const pr=atlas.role.get(pre), po=atlas.role.get(post);
    const isPlastic=pr==='KC'&&po==='MBON';
    if(isPlastic) plastic++;
    edges.push([pre,post,count]);
  }
  if(plastic<10) throw new Error(`subgrafo sem KC→MBON suficiente (${plastic}); confira se o arquivo é connections_princeton do FAFB v783`);
  graph={version:9,dataset:'FAFB v783',source,created:new Date().toISOString(),rowsScanned:rows,keptRows,keptSyn,edges};
  await dbPut(GRAPH_KEY,graph);
  await dbDelete(WEIGHT_KEY);
  buildRuntime();
  set('#importProgress',`${fmt(rows)} linhas · ${fmt(edges.length)} pares internos · ${fmt(plastic)} KC→MBON`);
  msg('sys',`Conectoma importado: ${fmt(edges.length)} conexões internas do mushroom body; ${fmt(plastic)} são KC→MBON plásticas.`);
}

function hash32(s, seed=2166136261){
  let h=seed>>>0;
  for(let i=0;i<s.length;i++){h=Math.imul(h^s.charCodeAt(i),16777619)>>>0;}
  return h>>>0;
}
function normalizeText(s){return String(s||'').toLowerCase().normalize('NFD').replace(/[\u0300-\u036f]/g,'').replace(/[^a-z0-9 ]+/g,' ').replace(/\s+/g,' ').trim();}
function encodeText(text){
  const s=normalizeText(text);
  const active=new Map();
  if(!s) return active;
  const grams=[];
  const padded='  '+s+'  ';
  for(let i=0;i<padded.length-2;i++) grams.push(padded.slice(i,i+3));
  grams.push('phrase:'+s);
  for(const g of grams){
    const h=hash32(g);
    for(let k=0;k<3;k++){
      const idx=(h + Math.imul(k+1,0x9e3779b1))>>>0;
      const kc=rt.kcIds[idx%rt.kcIds.length];
      active.set(kc,(active.get(kc)||0)+1);
    }
  }
  let max=1; for(const v of active.values()) if(v>max) max=v;
  for(const [k,v] of active) active.set(k,v/max);
  return active;
}

function buildRuntime(){
  if(!graph||!atlas) return;
  const mbonIds=[...atlas.mbon].sort();
  const kcIds=[...atlas.kc];
  const mbonIndex=new Map(mbonIds.map((x,i)=>[x,i]));
  const byKC=new Map();
  const plastic=[];
  let internal=0, kcm=0, dan=0;
  for(const [pre,post,count] of graph.edges){
    internal++;
    const pr=atlas.role.get(pre),po=atlas.role.get(post);
    if((pr==='PAM'||pr==='PPL')) dan++;
    if(pr==='KC'&&po==='MBON'){
      const e={pre,post,count:Number(count),gain:1,index:plastic.length};
      plastic.push(e); kcm++;
      if(!byKC.has(pre))byKC.set(pre,[]);
      byKC.get(pre).push(e);
    }
  }
  const pool=new Uint8Array(mbonIds.length);
  let a=0,b=0;
  for(let i=0;i<mbonIds.length;i++){
    pool[i]=hash32(mbonIds[i],0x811c9dc5)&1;
    pool[i]===0?a++:b++;
  }
  rt={mbonIds,kcIds,mbonIndex,byKC,plastic,pool,poolCounts:[a,b]};
  set('#graphState','READY');
  set('#edgeCount',fmt(internal));
  set('#plasticCount',fmt(kcm));
  set('#danEdgeCount',fmt(dan));
  set('#sourceName',graph.source||graph.dataset||'local');
  loadGains();
  drawMBON(null);
  status(`conectoma real pronto · ${fmt(kcm)} sinapses agregadas KC→MBON plásticas`);
}

async function loadGains(){
  if(!rt) return;
  try{
    const saved=await dbGet(WEIGHT_KEY);
    if(saved && saved.version===1 && Array.isArray(saved.keys) && saved.keys.length===rt.plastic.length){
      const map=new Map(saved.keys.map((k,i)=>[k,saved.gains[i]]));
      for(const e of rt.plastic){const v=map.get(e.pre+'|'+e.post);if(Number.isFinite(v))e.gain=v;}
      status('conectoma + plasticidade aprendida restaurados');
    }
  }catch(err){console.warn(err);}
}
async function saveGains(){
  if(!rt) return;
  const keys=new Array(rt.plastic.length), gains=new Array(rt.plastic.length);
  for(let i=0;i<rt.plastic.length;i++){const e=rt.plastic[i];keys[i]=e.pre+'|'+e.post;gains[i]=e.gain;}
  await dbPut(WEIGHT_KEY,{version:1,keys,gains,updated:new Date().toISOString()});
}

function infer(text){
  if(!rt) throw new Error('importe primeiro as conexões reais do FlyWire');
  const active=encodeText(text);
  const act=new Float64Array(rt.mbonIds.length);
  const eligible=[];
  let used=0;
  for(const [kc,x] of active){
    const edges=rt.byKC.get(kc); if(!edges) continue;
    for(const e of edges){
      const j=rt.mbonIndex.get(e.post); if(j==null) continue;
      const base=Math.log1p(e.count);
      const c=x*base*e.gain;
      act[j]+=c; used++;
      eligible.push([e,c]);
    }
  }
  let scores=[0,0], max=[0,0];
  for(let i=0;i<act.length;i++){
    const p=rt.pool[i];
    scores[p]+=act[i];
    if(act[i]>max[p])max[p]=act[i];
  }
  scores[0]=scores[0]/Math.max(1,rt.poolCounts[0]) + .15*max[0];
  scores[1]=scores[1]/Math.max(1,rt.poolCounts[1]) + .15*max[1];
  const winner=scores[0]>=scores[1]?0:1;
  const total=Math.abs(scores[0])+Math.abs(scores[1])+1e-9;
  const confidence=Math.abs(scores[0]-scores[1])/total;
  const token=winner===0?'SIM':'NÃO';
  lastTrace={text,active,act,eligible,winner,scores,token};
  set('#activeKC',fmt(active.size));
  set('#usedEdges',fmt(used));
  set('#scoreA',scores[0].toFixed(3));
  set('#scoreB',scores[1].toFixed(3));
  set('#confText',pct(confidence));
  $('#confidence').style.width=(confidence*100).toFixed(1)+'%';
  drawMBON(act);
  return{token,confidence,scores,active:active.size,used};
}

function applyFeedback(sign){
  if(!lastTrace||!rt) return;
  const winner=lastTrace.winner;
  const relevant=lastTrace.eligible.filter(([e])=>rt.pool[rt.mbonIndex.get(e.post)]===winner);
  if(!relevant.length){msg('sys','Sem arestas elegíveis nessa resposta.');return;}
  let max=0;for(const [,c] of relevant)if(c>max)max=c;
  const eta=sign>0?.045:.065;
  let changed=0;
  for(const [e,c] of relevant){
    const eligibility=c/(max||1);
    if(sign>0)e.gain=Math.min(4, e.gain*(1+eta*eligibility));
    else e.gain=Math.max(.18, e.gain*(1-eta*eligibility));
    changed++;
  }
  set('#lastModulator',sign>0?'PAM-like +':'PPL-like −');
  msg('sys',`${sign>0?'👍 PAM-like recompensa':'👎 PPL-like punição'}: ${fmt(changed)} conexões KC→MBON elegíveis moduladas. Nenhum alvo textual foi injetado.`);
  clearTimeout(saveTimer); saveTimer=setTimeout(()=>saveGains().catch(console.error),800);
}

function drawMBON(act){
  const cv=$('#brain'), cx=cv.getContext('2d');
  const w=cv.width,h=cv.height;
  cx.fillStyle='#080d14';cx.fillRect(0,0,w,h);
  if(!rt){cx.fillStyle='#93a2b8';cx.font='16px system-ui';cx.fillText('aguardando conectoma real…',14,28);return;}
  const n=rt.mbonIds.length,bw=w/n;
  let mx=1;
  if(act)for(const x of act)if(x>mx)mx=x;
  for(let i=0;i<n;i++){
    const v=act?act[i]/mx:0;
    cx.fillStyle=rt.pool[i]===0?'rgba(117,230,165,.82)':'rgba(120,167,255,.82)';
    cx.fillRect(i*bw+1,h-18-v*(h-40),Math.max(1,bw-1),v*(h-40));
  }
  cx.fillStyle='#93a2b8';cx.font='13px system-ui';
  cx.fillText('MBONs reais · verde = canal SIM · azul = canal NÃO',12,18);
}

async function loadSavedGraph(){
  try{graph=await dbGet(GRAPH_KEY);if(graph){buildRuntime();set('#importProgress','subgrafo restaurado do IndexedDB');msg('sys','Subgrafo FlyWire restaurado deste navegador.');}}
  catch(err){console.warn(err);}
}

async function importFile(file){
  if(!file) return;
  const gz=file.name.toLowerCase().endsWith('.gz');
  await processConnectionStream(file.stream(),gz,'Codex '+file.name);
}
async function importOfficial(){
  $('#official').disabled=true;
  try{
    set('#importProgress','conectando ao Codex…');
    const res=await fetch(CODEX_URL,{mode:'cors'});
    if(!res.ok)throw new Error('HTTP '+res.status);
    const ct=(res.headers.get('content-type')||'').toLowerCase();
    const cd=(res.headers.get('content-disposition')||'').toLowerCase();
    const ce=(res.headers.get('content-encoding')||'').toLowerCase();
    const gz=!ce && (ct.includes('gzip')||cd.includes('.gz'));
    await processConnectionStream(res.body,gz,'Codex connections_princeton FAFB v783');
  }catch(err){
    console.error(err);
    msg('sys','Carga automática do Codex falhou ('+err.message+'). Baixe connections_princeton.csv.gz no Codex e selecione o arquivo abaixo; nada precisa ser instalado.');
    status('use o importador de arquivo oficial');
  }finally{$('#official').disabled=false;}
}

function exportGraph(){
  if(!graph)return;
  const payload={...graph,note:'Filtered FlyWire FAFB v783 mushroom-body subgraph; original source Codex connections_princeton.'};
  const blob=new Blob([JSON.stringify(payload)],{type:'application/json'});
  const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='flywire-mushroom-body-v783.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),5000);
}

$('#send').onclick=()=>{
  const text=$('#input').value.trim(); if(!text)return;
  $('#input').value='';msg('user',text);
  try{const r=infer(text);msg('fly',`${r.token} · confiança ${(r.confidence*100).toFixed(1)}%`);status(`saída emergiu de ${fmt(r.active)} KCs ativos e ${fmt(r.used)} arestas KC→MBON reais`);}catch(err){msg('sys',err.message);status(err.message);}
};
$('#input').addEventListener('keydown',e=>{if(e.key==='Enter')$('#send').click();});
$('#reward').onclick=()=>applyFeedback(1);
$('#punish').onclick=()=>applyFeedback(-1);
$('#official').onclick=()=>importOfficial();
$('#file').onchange=e=>importFile(e.target.files?.[0]).catch(err=>{console.error(err);msg('sys','Erro ao importar: '+err.message);set('#graphState','ERRO');});
$('#export').onclick=exportGraph;
$('#save').onclick=()=>saveGains().then(()=>msg('sys','Plasticidade salva no navegador.'));
$('#clearLearning').onclick=async()=>{if(!confirm('Zerar apenas a plasticidade aprendida? O conectoma real será mantido.'))return;await dbDelete(WEIGHT_KEY);if(rt)for(const e of rt.plastic)e.gain=1;msg('sys','Ganhos plásticos zerados; topologia FlyWire mantida.');};
$('#clearGraph').onclick=async()=>{if(!confirm('Remover o subgrafo importado deste navegador?'))return;await dbDelete(GRAPH_KEY);await dbDelete(WEIGHT_KEY);graph=null;rt=null;lastTrace=null;set('#graphState','AUSENTE');set('#edgeCount','0');set('#plasticCount','0');set('#danEdgeCount','0');drawMBON(null);msg('sys','Subgrafo local removido.');};

(async()=>{
  try{await loadAtlas();await loadSavedGraph();}
  catch(err){console.error(err);set('#atlasState','ERRO');status(err.message);msg('sys','Falha ao iniciar: '+err.message);}
})();
