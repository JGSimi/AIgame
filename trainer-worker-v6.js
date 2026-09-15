const F=896,H=80,MAXV=640,NEG=12,LR=.008;
let vocab=['<eos>','<unk>'],vmap=new Map(vocab.map((x,i)=>[x,i]));
let feat=new Float32Array(F*H),tok=new Float32Array(MAXV*H),out=new Float32Array(MAXV*H),bias=new Float32Array(MAXV),bh=new Float32Array(H);
let pairs=[],userCount=0,running=true,steps=0,stage='auto',autoLevel=0,sliceMs=440;
let lastReport=performance.now(),reportSteps=0,lossEMA=0,trainAccEMA=0,lastSnapshot=0,nextEval=5000;
let metrics={overall:0,paraphrase:0,compose:0,context:0,exact:0};

const FACTS=[
 ['ceu','azul'],['grama','verde'],['banana','amarela'],['morango','vermelho'],['carvao','preto'],['neve','branca'],
 ['fogo','quente'],['gelo','frio'],['mel','doce'],['limao','azedo'],['pedra','dura'],['algodao','macio']
];
const PEOPLE=['ana','bia','clara','davi','enzo','flora','gabriel','heloisa'];
const THINGS=['maca','banana','uva','mel','luz','musica','chuva','cafe'];
const TRAIN_PARAPHRASE=[
 [['oi','ola','e ai','bom dia','boa tarde'],'oi como voce esta'],
 [['como voce esta','como vai','tudo bem com voce'],'estou bem e aprendendo'],
 [['quem e voce','o que voce e','me diga quem voce e'],'eu sou uma mosca neural aprendendo a conversar'],
 [['o que voce quer','qual seu objetivo','o que voce pretende'],'eu quero aprender e conversar melhor'],
 [['voce aprende','consegue aprender','esta aprendendo'],'sim eu aprendo com treino exemplos e correcoes'],
 [['podemos conversar','quer conversar','fale comigo'],'sim podemos conversar'],
 [['obrigado','valeu','muito obrigado'],'de nada'],
 [['tchau','ate logo','ate mais'],'ate mais'],
 [['voce sente dor','voce sente medo','voce sente emocoes'],'eu nao sinto isso como um animal real'],
 [['voce tem consciencia','voce e consciente'],'nao eu sou uma rede neural experimental']
];
const HELD_PARAPHRASE=[
 ['salve mosca','oi como voce esta'],['como andam as coisas','estou bem e aprendendo'],['quem esta falando comigo','eu sou uma mosca neural aprendendo a conversar'],
 ['pra que voce serve','eu quero aprender e conversar melhor'],['da pra voce aprender coisas novas','sim eu aprendo com treino exemplos e correcoes'],
 ['bora trocar ideia','sim podemos conversar'],['agradecido','de nada'],['falou ate outra hora','ate mais'],
 ['voce realmente sente alguma emocao','eu nao sinto isso como um animal real'],['existe consciencia ai dentro','nao eu sou uma rede neural experimental']
];
function rnd(a=.065){return(Math.random()*2-1)*a}
for(let i=0;i<feat.length;i++)feat[i]=rnd();for(let i=0;i<tok.length;i++)tok[i]=rnd();for(let i=0;i<out.length;i++)out[i]=rnd();
function norm(s){return String(s||'').toLowerCase().normalize('NFD').replace(/[\u0300-\u036f]/g,'').replace(/([!?.,:])/g,' $1 ').replace(/\s+/g,' ').trim()}
function words(s){return norm(s).split(' ').filter(Boolean)}
function ensure(w){if(vmap.has(w))return vmap.get(w);if(vocab.length>=MAXV)return 1;const id=vocab.length;vocab.push(w);vmap.set(w,id);for(let h=0;h<H;h++){tok[id*H+h]=rnd();out[id*H+h]=rnd()}return id}
function ids(s,add=true){return words(s).map(w=>add?ensure(w):(vmap.get(w)??1))}
function feats(s){const n=norm(s),set=new Set(),ws=n.split(' ').filter(Boolean);for(let wi=0;wi<ws.length;wi++){const w=ws[wi];let h=2166136261;for(let i=0;i<w.length;i++)h=Math.imul(h^w.charCodeAt(i),16777619)>>>0;set.add(h%F);set.add(((h^Math.imul(wi+1,2654435761))>>>0)%F);if(wi>0){let p=2166136261,bi=ws[wi-1]+'_'+w;for(let i=0;i<bi.length;i++)p=Math.imul(p^bi.charCodeAt(i),16777619)>>>0;set.add(p%F)}}for(let i=0;i<n.length-1;i++)set.add(((n.charCodeAt(i)*131+n.charCodeAt(i+1)*17+i*7)>>>0)%F);return[...set].slice(0,64)}
function sig(x){return x>20?1:x<-20?0:1/(1+Math.exp(-x))}
function hidden(fs,p1,p2,p3,p4){const h=new Float32Array(H),scale=1/Math.max(1,fs.length);for(let j=0;j<H;j++){let z=bh[j]+tok[p1*H+j]+.52*tok[p2*H+j]+.27*tok[p3*H+j]+.12*tok[p4*H+j];for(const f of fs)z+=feat[f*H+j]*scale;h[j]=Math.tanh(z)}return h}
function score(h,t){let s=bias[t],o=t*H;for(let j=0;j<H;j++)s+=out[o+j]*h[j];return s}
function trainPair(p){const fs=feats(p[0]),ys=ids(p[1],true);ys.push(0);let p1=0,p2=0,p3=0,p4=0,total=0,correct=0;for(const y of ys){const h=hidden(fs,p1,p2,p3,p4),dh=new Float32Array(H),sy=score(h,y),py=sig(sy),eosScale=y===0?.18:1,gy=(py-1)*eosScale;total+=-eosScale*Math.log(Math.max(1e-6,py));let oy=y*H;for(let j=0;j<H;j++){dh[j]+=gy*out[oy+j];out[oy+j]-=LR*gy*h[j]}bias[y]-=LR*gy;
 let bestNeg=-1e9;for(let n=0;n<NEG;n++){let q=2+(Math.random()*Math.max(1,vocab.length-2)|0);if(q===y)q=2+((q+7)%Math.max(1,vocab.length-2));const sq=score(h,q);if(sq>bestNeg)bestNeg=sq;const pq=sig(sq),gq=pq,oq=q*H;total+=-.08*Math.log(Math.max(1e-6,1-pq));for(let j=0;j<H;j++){dh[j]+=.08*gq*out[oq+j];out[oq+j]-=LR*.08*gq*h[j]}bias[q]-=LR*.08*gq}
 if(y===0||sy>bestNeg)correct++;const sc=1/Math.max(1,fs.length);for(let j=0;j<H;j++){const dz=dh[j]*(1-h[j]*h[j]);bh[j]-=LR*.14*dz;tok[p1*H+j]-=LR*.50*dz;tok[p2*H+j]-=LR*.27*dz;tok[p3*H+j]-=LR*.14*dz;tok[p4*H+j]-=LR*.07*dz;for(const f of fs)feat[f*H+j]-=LR*.28*dz*sc}p4=p3;p3=p2;p2=p1;p1=y}
 return{loss:total/ys.length,acc:correct/ys.length}}
function logits(h){const a=new Float32Array(vocab.length);let m=-1e9;for(let i=0;i<a.length;i++){a[i]=score(h,i);if(a[i]>m)m=a[i]}let s=0;for(let i=0;i<a.length;i++){a[i]=Math.exp((a[i]-m)/.66);s+=a[i]}for(let i=0;i<a.length;i++)a[i]/=s||1;return a}
function detok(a){return a.join(' ').replace(/ \?/g,'?').replace(/ !/g,'!').replace(/ ,/g,',').replace(/ \./g,'.').replace(/ :/g,':')}
function chooseToken(p,res,k,creative){let top=[];for(let i=2;i<p.length;i++){let s=p[i],w=vocab[i],repeat=0;for(const x of res)if(x===w)repeat++;if(repeat)s*=repeat===1?.42:.08;if(res.length&&res[res.length-1]===w)s*=.25;top.push([i,s])}top.sort((a,b)=>b[1]-a[1]);const eos=p[0]||0;if(k>0&&eos>(top[0]?.[1]||0)*.82)return 0;if(!creative||top.length<2)return top[0]?.[0]||2;const kk=Math.min(5,top.length);let sum=0;for(let i=0;i<kk;i++)sum+=Math.pow(top[i][1],1/.78);let r=Math.random()*sum;for(let i=0;i<kk;i++){r-=Math.pow(top[i][1],1/.78);if(r<=0)return top[i][0]}return top[0][0]}
function generate(prompt,max=22,creative=false){const fs=feats(prompt);let p1=0,p2=0,p3=0,p4=0,res=[],conf=0,hlast=[];for(let k=0;k<max;k++){const h=hidden(fs,p1,p2,p3,p4),p=logits(h);hlast=Array.from(h);const best=chooseToken(p,res,k,creative);if(best===0){if(res.length)break;continue}const w=vocab[best];if(!w||w==='<unk>')continue;res.push(w);conf+=p[best]||0;p4=p3;p3=p2;p2=p1;p1=best}if(!res.length)res.push('aprendendo');return{text:detok(res),confidence:conf/Math.max(1,res.length),hidden:hlast}}
function tokenScore(a,b){const aa=words(a),bb=words(b);if(!bb.length)return 0;let hit=0;for(let i=0;i<Math.min(aa.length,bb.length);i++)if(aa[i]===bb[i])hit++;return hit/Math.max(aa.length,bb.length)}
function exact(a,b){return norm(a)===norm(b)?1:0}
function staticExample(){if(!pairs.length)return null;return pairs[Math.random()*Math.max(1,pairs.length-userCount)|0]}
function paraphraseExample(held=false){const src=held?HELD_PARAPHRASE:TRAIN_PARAPHRASE;const g=src[Math.random()*src.length|0];if(held)return[g[0],g[1]];return[g[0][Math.random()*g[0].length|0],g[1]]}
function composeExample(held=false){const f=FACTS[(Math.random()*FACTS.length)|0],sub=f[0],val=f[1];const templates=held?[
 `me diga sem repetir a pergunta : ${sub} tem qual caracteristica`,
 `considerando o que voce sabe , como voce descreve ${sub}`,
 `qual palavra combina com ${sub} nesse contexto`
]:[
 `qual a caracteristica de ${sub}`,
 `como e ${sub}`,
 `me diga algo sobre ${sub}`,
 `o que combina com ${sub}`
];return[templates[Math.random()*templates.length|0],`${sub} e ${val}`]}
function contextExample(held=false){const name=PEOPLE[Math.random()*PEOPLE.length|0],thing=THINGS[Math.random()*THINGS.length|0];if(Math.random()<.5){const prompts=held?[
 `usuario : meu nome e ${name} | mosca : prazer em conhecer voce | usuario : consegue lembrar meu nome`,
 `usuario : pode me chamar de ${name} | mosca : certo | usuario : quem eu disse que sou`
]:[
 `usuario : meu nome e ${name} | mosca : prazer ${name} | usuario : qual e meu nome`,
 `usuario : eu me chamo ${name} | mosca : entendi | usuario : como eu me chamo`
];return[prompts[Math.random()*prompts.length|0],`seu nome e ${name}`]}const prompts=held?[
 `usuario : uma coisa que eu curto e ${thing} | mosca : legal | usuario : voce lembra do que eu curto`,
 `usuario : ${thing} e algo de que eu gosto | mosca : entendi | usuario : qual preferencia eu mencionei`
]:[
 `usuario : eu gosto de ${thing} | mosca : entendi | usuario : do que eu gosto`,
 `usuario : minha preferencia e ${thing} | mosca : certo | usuario : qual e minha preferencia`
];return[prompts[Math.random()*prompts.length|0],`voce gosta de ${thing}`]}
function negativeExample(){const a=Math.random()<.5?['pedra','fogo','parede','veneno'][Math.random()*4|0]:['comida','luz','agua'][Math.random()*3|0];if(['pedra','fogo','parede','veneno'].includes(a))return[`isso e comida : ${a}`,'nao isso nao e comida'];return[`isso e algo positivo : ${a}`,'sim isso pode ser positivo']}
function sampleTrain(){if(userCount>0&&Math.random()<.22)return pairs[pairs.length-userCount+(Math.random()*userCount|0)];if(stage==='basic')return Math.random()<.7?staticExample():paraphraseExample(false);if(stage==='qa')return Math.random()<.5?composeExample(false):negativeExample();if(stage==='conversation')return Math.random()<.65?contextExample(false):paraphraseExample(false);autoLevel=steps>180000?3:steps>70000?2:steps>18000?1:0;const r=Math.random();if(autoLevel===0)return r<.6?staticExample():paraphraseExample(false);if(autoLevel===1)return r<.35?staticExample():r<.7?paraphraseExample(false):composeExample(false);if(autoLevel===2)return r<.22?staticExample():r<.50?paraphraseExample(false):r<.78?composeExample(false):contextExample(false);return r<.16?staticExample():r<.38?paraphraseExample(false):r<.64?composeExample(false):r<.90?contextExample(false):negativeExample()}
function augmentPrompt(s){let x=String(s);if(Math.random()<.12)x='ei , '+x;if(Math.random()<.10)x='por favor , '+x;if(Math.random()<.18)x=x.replace(/[?!.]/g,'');if(Math.random()<.08)x=x.replace(/voce/g,'vc');return x}
function fixedValidation(){const out=[];for(const x of HELD_PARAPHRASE)out.push({g:'paraphrase',p:x});for(let i=0;i<24;i++)out.push({g:'compose',p:composeExample(true)});for(let i=0;i<24;i++)out.push({g:'context',p:contextExample(true)});return out}
let VAL=[];
function evalModel(){if(!VAL.length)VAL=fixedValidation();let sums={paraphrase:[0,0,0],compose:[0,0,0],context:[0,0,0]},allTok=0,allEx=0,n=0;for(const item of VAL){const g=generate(item.p[0],22,false).text,tokS=tokenScore(g,item.p[1]),ex=exact(g,item.p[1]);const a=sums[item.g];a[0]+=tokS;a[1]+=ex;a[2]++;allTok+=tokS;allEx+=ex;n++}const p=sums.paraphrase,c=sums.compose,x=sums.context;return{overall:allTok/n,exact:allEx/n,paraphrase:p[0]/p[2],compose:c[0]/c[2],context:x[0]/x[2]}}
function pack(a){const u=new Uint8Array(a.buffer,a.byteOffset,a.byteLength);let s='',chunk=0x8000;for(let i=0;i<u.length;i+=chunk)s+=String.fromCharCode(...u.subarray(i,i+chunk));return btoa(s)}
function unpack(str,target){try{const bin=atob(str),u=new Uint8Array(bin.length);for(let i=0;i<bin.length;i++)u[i]=bin.charCodeAt(i);const f=new Float32Array(u.buffer);target.set(f.subarray(0,target.length));return true}catch{return false}}
function snapshot(){return{version:6,vocab,feat:pack(feat),tok:pack(tok.subarray(0,vocab.length*H)),out:pack(out.subarray(0,vocab.length*H)),bias:pack(bias.subarray(0,vocab.length)),bh:pack(bh),steps,lossEMA,trainAccEMA,metrics}}
function restore(s){try{if(!s||s.version!==6||!Array.isArray(s.vocab))return;vocab=s.vocab.slice(0,MAXV);vmap=new Map(vocab.map((x,i)=>[x,i]));unpack(s.feat,feat);if(s.tok){const temp=new Float32Array(vocab.length*H);if(unpack(s.tok,temp))tok.set(temp)}if(s.out){const temp=new Float32Array(vocab.length*H);if(unpack(s.out,temp))out.set(temp)}if(s.bias){const temp=new Float32Array(vocab.length);if(unpack(s.bias,temp))bias.set(temp)}unpack(s.bh,bh);steps=s.steps||0;lossEMA=s.lossEMA||0;trainAccEMA=s.trainAccEMA||0;metrics=s.metrics||metrics;nextEval=steps+5000}catch{}}
function trainOne(){const p=sampleTrain();if(!p)return false;const r=trainPair([augmentPrompt(p[0]),p[1]]);lossEMA=steps?lossEMA*.997+r.loss*.003:r.loss;trainAccEMA=steps?trainAccEMA*.997+r.acc*.003:r.acc;steps++;reportSteps++;return true}
function loop(){if(!running){setTimeout(loop,30);return}const start=performance.now();while(running&&performance.now()-start<sliceMs){if(!trainOne())break}const now=performance.now();if(steps>=nextEval){metrics=evalModel();nextEval=steps+Math.max(7000,Math.floor(reportSteps*1.2))}if(now-lastReport>400){const speed=reportSteps/((now-lastReport)/1000);postMessage({type:'stats',steps,speed,loss:lossEMA,trainAcc:trainAccEMA,...metrics,vocab:vocab.length,pairs:pairs.length+TRAIN_PARAPHRASE.length+FACTS.length,level:autoLevel});reportSteps=0;lastReport=now;if(steps-lastSnapshot>60000){lastSnapshot=steps;postMessage({type:'snapshot',state:snapshot()})}}setTimeout(loop,0)}

onmessage=e=>{const m=e.data||{};if(m.type==='init'){pairs=m.pairs||[];userCount=m.userCount||0;for(const p of pairs){ids(p[0],true);ids(p[1],true)}for(const g of TRAIN_PARAPHRASE){for(const q of g[0])ids(q,true);ids(g[1],true)}for(const x of HELD_PARAPHRASE){ids(x[0],true);ids(x[1],true)}for(const [a,b] of FACTS){ids(a,true);ids(b,true)}for(const n of PEOPLE)ensure(n);for(const t of THINGS)ensure(t);restore(m.state);running=m.running!==false;stage=m.stage||'auto';sliceMs=Math.max(80,Math.min(700,m.sliceMs||440));VAL=fixedValidation();postMessage({type:'ready',vocab:vocab.length,pairs:pairs.length,steps,loss:lossEMA,trainAcc:trainAccEMA,...metrics});loop()}else if(m.type==='run'){running=!!m.running}else if(m.type==='stage'){stage=m.stage||'auto'}else if(m.type==='speed'){sliceMs=Math.max(80,Math.min(700,m.sliceMs||440))}else if(m.type==='infer'){postMessage({type:'infer',id:m.id,...generate(m.prompt,24,true)})}else if(m.type==='teach'){const p=[m.prompt,m.reply];pairs.push(p);userCount++;ids(p[0],true);ids(p[1],true);for(let i=0;i<220;i++)trainPair(p);postMessage({type:'taught',vocab:vocab.length,pairs:pairs.length})}else if(m.type==='positive'){const p=[m.prompt,m.reply];pairs.push(p);userCount++;ids(p[0],true);ids(p[1],true);for(let i=0;i<55;i++)trainPair(p)}else if(m.type==='snapshot'){postMessage({type:'snapshot',state:snapshot()})}else if(m.type==='evaluate'){metrics=evalModel();postMessage({type:'evaluation',...metrics})}};
