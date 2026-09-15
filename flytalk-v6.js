const $=s=>document.querySelector(s);
const BASE_PAIRS=[
['oi','oi como voce esta'],['ola','oi como voce esta'],['e ai','oi como voce esta'],['bom dia','bom dia como voce esta'],['boa tarde','boa tarde'],['boa noite','boa noite'],
['como voce esta','estou bem e aprendendo'],['tudo bem com voce','sim estou bem e aprendendo'],['como vai','estou bem e aprendendo'],
['quem e voce','eu sou uma mosca neural aprendendo a conversar'],['o que voce e','eu sou uma rede neural experimental'],['qual e seu nome','pode me chamar de fly'],
['voce aprende','sim eu aprendo com treino exemplos e correcoes'],['como voce aprende','eu aprendo ajustando meus pesos com exemplos e correcoes'],['voce esta aprendendo','sim estou aprendendo agora'],
['o que voce quer','eu quero aprender e conversar melhor'],['qual seu objetivo','meu objetivo e aprender a conversar com voce'],['podemos conversar','sim podemos conversar'],
['voce quer comida','sim eu quero comida'],['o que e comida','comida e algo associado a alimentacao'],['comida e boa','sim comida pode ser algo positivo'],
['o que e perigo','perigo e um sinal para evitar algo ruim'],['isso e perigoso','se e perigoso eu devo evitar'],['voce sente perigo','eu posso aprender sinais associados a perigo'],
['o que e luz','luz e um estimulo visual'],['o que e escuro','escuro e ausencia ou pouca luz'],['tem luz','sim existe luz'],['esta escuro','sim esta escuro'],
['isso doi','dor e um sinal associado a algo ruim'],['voce sente dor','eu nao sinto dor real mas posso aprender o conceito'],
['voce sente emocoes','eu nao sinto emocoes como um animal real'],['voce tem medo','eu nao sinto medo como um animal real'],['voce tem consciencia','nao eu sou uma rede neural experimental'],
['voce lembra de mim','eu consigo usar um pouco do contexto recente da conversa'],['voce entende contexto','eu estou treinando para usar contexto de varias mensagens'],
['o que voce sabe','eu sei padroes que aprendi durante o treino'],['voce sabe tudo','nao eu ainda tenho muito para aprender'],['voce e inteligente','minha inteligencia depende do que minha rede consegue aprender'],
['voce consegue conversar naturalmente','eu estou treinando para formar respostas mais naturais'],['por que voce erra','porque ainda estou aprendendo e generalizando'],
['quem te ensina','voce e os exemplos de treino me ensinam'],['o que acontece quando eu te corrijo','sua correcao entra no meu treinamento'],['o treino ajuda','sim mas preciso de variedade e generalizacao'],
['obrigado','de nada'],['valeu','de nada'],['muito obrigado','de nada'],['tchau','ate mais'],['ate logo','ate mais'],
['fale comigo','estou aqui para conversar'],['me responda','estou aqui para responder'],['responda','estou respondendo'],['vamos conversar','sim vamos conversar'],
['qual a cor do ceu','o ceu e azul'],['qual a cor da grama','a grama e verde'],['como e o gelo','gelo e frio'],['como e o fogo','fogo e quente'],
['pedra e comida','nao pedra nao e comida'],['parede e comida','nao parede nao e comida'],['fogo e comida','nao isso nao e comida'],
['qual e o contrario de sim','o contrario de sim e nao'],['qual e o contrario de bom','o contrario de bom e ruim'],
['o que significa aprender','aprender e mudar meus parametros a partir de exemplos'],['o que significa loss','loss mede quanto o modelo ainda esta errando'],
['o que significa generalizacao','generalizacao e responder bem a exemplos diferentes dos usados no treino'],
['o que e validacao','validacao testa exemplos separados dos exemplos usados para treinar'],
['voce pode melhorar','sim minhas respostas podem melhorar com treino e correcoes'],['quer aprender mais','sim quero aprender mais'],['vamos treinar','sim vamos treinar']
];

let userPairs=[];try{userPairs=JSON.parse(localStorage.getItem('flytalk-user-pairs-v6')||localStorage.getItem('flytalk-user-pairs-v5')||'[]')}catch{}
let history=[],lastPrompt='',lastContextPrompt='',lastReply='',lastConfidence=0,lastHidden=[],worker=null,inferId=0;
let running=localStorage.getItem('flytalk-running-v6')!=='0';
const mobile=matchMedia('(max-width:650px)').matches;

function addMsg(kind,text){const el=document.createElement('div');el.className='msg '+kind;const who=document.createElement('div');who.className='who';who.textContent=kind==='user'?'VOCE':kind==='fly'?'MOSCA':'SISTEMA';if(kind!=='sys')el.appendChild(who);el.appendChild(document.createTextNode(text));$('#chat').appendChild(el);while($('#chat').children.length>80)$('#chat').firstChild.remove();$('#chat').scrollTop=$('#chat').scrollHeight}
function saveUserPairs(){localStorage.setItem('flytalk-user-pairs-v6',JSON.stringify(userPairs.slice(-400)))}
function pct(v){return Number.isFinite(v)?(v*100).toFixed(v>=.995?0:1)+'%':'—'}
function num(v){return Number(v||0).toLocaleString('pt-BR')}
function setMetric(id,v){const el=$(id);if(el)el.textContent=v}
function powerMs(){const p=$('#power')?.value||'max';return p==='max'?(mobile?340:560):p==='high'?(mobile?220:340):120}
function makeContext(text){if(!history.length)return text;const recent=history.slice(-6).map(x=>(x.role==='user'?'usuario':'mosca')+' : '+x.text).join(' | ');return recent+' | usuario : '+text}
function updateTrainButton(){setMetric('#train',running?'⏸ Pausar treino':'▶ Retomar treino');$('#train').classList.toggle('running',running);setMetric('#state',running?'TREINANDO V6':'PAUSADO');localStorage.setItem('flytalk-running-v6',running?'1':'0')}
function setQuality(id,v){const el=$(id);if(!el)return;el.textContent=pct(v);const card=el.parentElement;card.classList.toggle('ok',v>=.85);card.classList.toggle('warn',v>=.55&&v<.85)}
function syncMetrics(m){
 setMetric('#steps',num(m.steps));setMetric('#liveSteps',num(m.steps));setMetric('#speed',num(Math.round(m.speed||0)));setMetric('#liveSpeed',num(Math.round(m.speed||0)));
 setMetric('#loss',Number.isFinite(m.loss)?m.loss.toFixed(3):'—');setMetric('#liveLoss',Number.isFinite(m.loss)?m.loss.toFixed(3):'—');
 setQuality('#trainAcc',m.trainAcc||0);setQuality('#overall',m.overall||0);setQuality('#valExact',m.exact||0);setQuality('#paraphrase',m.paraphrase||0);setQuality('#compose',m.compose||0);setQuality('#context',m.context||0);
 setQuality('#liveGeneral',m.overall||0);setQuality('#liveContext',m.context||0);setQuality('#liveExact',m.exact||0);
 setMetric('#vocab',num(m.vocab));setMetric('#pairs',num(m.pairs));
 const q=Math.max(0,Math.min(1,m.overall||0));$('#progress').style.width=(q*100)+'%';
 setMetric('#trainInfo',`${num(m.steps)} exemplos · nivel ${1+(m.level||0)}/4 · geral ${pct(m.overall||0)} · ${num(Math.round(m.speed||0))}/s`);
 setMetric('#event',`generalizacao ${pct(m.overall||0)} · parafrase ${pct(m.paraphrase||0)} · composicao ${pct(m.compose||0)} · contexto ${pct(m.context||0)}`);
}
function startWorker(){if(worker)worker.terminate();worker=new Worker('trainer-worker-v6.js?v=6');let saved=null;try{saved=JSON.parse(localStorage.getItem('flytalk-model-v6')||'null')}catch{}
 worker.onmessage=e=>{const m=e.data||{};
  if(m.type==='ready'){syncMetrics({...m,speed:0});updateTrainButton()}
  else if(m.type==='stats'){syncMetrics(m)}
  else if(m.type==='snapshot'){try{localStorage.setItem('flytalk-model-v6',JSON.stringify(m.state))}catch(err){setMetric('#event','nao foi possivel salvar snapshot local: '+err.message)}}
  else if(m.type==='infer'){lastReply=(m.text||'').trim()||'ainda estou aprendendo a responder';lastConfidence=m.confidence||0;lastHidden=m.hidden||[];addMsg('fly',lastReply);history.push({role:'fly',text:lastReply});history=history.slice(-10);$('#confidence').style.width=Math.min(100,lastConfidence*100)+'%';setMetric('#confText',pct(lastConfidence));setMetric('#event','resposta v6 gerada com contexto recente');drawHidden()}
  else if(m.type==='taught'){setMetric('#vocab',num(m.vocab));setMetric('#pairs',num(m.pairs));setMetric('#userPairs',num(userPairs.length));addMsg('sys','Correcao adicionada ao replay v6.')}
  else if(m.type==='evaluation'){setQuality('#overall',m.overall||0);setQuality('#valExact',m.exact||0);setQuality('#paraphrase',m.paraphrase||0);setQuality('#compose',m.compose||0);setQuality('#context',m.context||0);setQuality('#liveGeneral',m.overall||0);setQuality('#liveContext',m.context||0);setQuality('#liveExact',m.exact||0);setMetric('#event',`teste manual: geral ${pct(m.overall||0)} · exata ${pct(m.exact||0)}`)}
 };
 worker.postMessage({type:'init',pairs:[...BASE_PAIRS,...userPairs],userCount:userPairs.length,state:saved,running,stage:$('#curriculum').value,sliceMs:powerMs()});
}
function send(){const text=$('#input').value.trim();if(!text||!worker)return;$('#input').value='';lastPrompt=text;lastContextPrompt=makeContext(text);addMsg('user',text);history.push({role:'user',text});history=history.slice(-10);worker.postMessage({type:'infer',id:++inferId,prompt:lastContextPrompt})}
$('#send').onclick=send;$('#input').addEventListener('keydown',e=>{if(e.key==='Enter')send()});
$('#train').onclick=()=>{running=!running;worker?.postMessage({type:'run',running});updateTrainButton()};
$('#curriculum').onchange=()=>{worker?.postMessage({type:'stage',stage:$('#curriculum').value});setMetric('#stageInfo',$('#curriculum').value==='auto'?'Mistura parafrase, composicao e memoria de contexto conforme o treino avanca.':'Foco alterado sem reiniciar os pesos.')};
$('#power').onchange=()=>{worker?.postMessage({type:'speed',sliceMs:powerMs()});setMetric('#powerInfo',`janela ${powerMs()} ms · worker em saturacao`)};
$('#good').onclick=()=>{if(!lastPrompt||!lastReply)return;worker.postMessage({type:'positive',prompt:lastContextPrompt||lastPrompt,reply:lastReply});addMsg('sys','👍 Resposta reforcada e adicionada ao replay.')};
$('#bad').onclick=()=>{if(!lastPrompt)return;addMsg('sys','👎 Marcada como ruim. Escreva abaixo a resposta que voce gostaria de ensinar.');$('#teacher').focus()};
$('#teach').onclick=()=>{const reply=$('#teacher').value.trim();if(!lastPrompt||!reply)return;$('#teacher').value='';const prompt=lastContextPrompt||lastPrompt;userPairs.push([prompt,reply]);userPairs=userPairs.slice(-400);saveUserPairs();worker.postMessage({type:'teach',prompt,reply})};
$('#save').onclick=()=>worker?.postMessage({type:'snapshot'});
$('#evaluate').onclick=()=>worker?.postMessage({type:'evaluate'});
$('#clearContext').onclick=()=>{history=[];addMsg('sys','Contexto da conversa limpo. O modelo treinado nao foi alterado.')};
$('#reset').onclick=()=>{if(!confirm('Zerar apenas os pesos da v6? Suas correcoes ensinadas serao preservadas.'))return;localStorage.removeItem('flytalk-model-v6');history=[];startWorker();addMsg('sys','Pesos v6 reiniciados; suas correcoes foram preservadas.')};
setMetric('#userPairs',num(userPairs.length));setMetric('#powerInfo',`janela ${powerMs()} ms · worker em saturacao`);updateTrainButton();startWorker();

const cv=$('#brain'),cx=cv.getContext('2d');function drawHidden(){cx.fillStyle='#080d14';cx.fillRect(0,0,cv.width,cv.height);if(!lastHidden.length){cx.fillStyle='#93a2b8';cx.font='16px system-ui';cx.fillText('aguardando uma resposta…',14,24);return}const n=lastHidden.length,bw=cv.width/n;for(let i=0;i<n;i++){const v=lastHidden[i],h=Math.abs(v)*(cv.height*.78);cx.fillStyle=v>=0?'rgba(117,230,165,.78)':'rgba(255,127,139,.72)';cx.fillRect(i*bw+1,cv.height*.5-(v>0?h:0),Math.max(1,bw-1),h)}cx.strokeStyle='#36445a';cx.beginPath();cx.moveTo(0,cv.height*.5);cx.lineTo(cv.width,cv.height*.5);cx.stroke();cx.fillStyle='#93a2b8';cx.font='14px system-ui';cx.fillText('80 unidades ocultas · ultima resposta',12,20)}drawHidden();
setInterval(()=>worker?.postMessage({type:'snapshot'}),35000);
