'use strict';
const $ = id => document.getElementById(id);
const labels = I18n.labels({alive:'存活奖励',angle:'倾斜惩罚',position:'偏移惩罚',failure:'失败惩罚'});
const hints = I18n.labels({alive:'每走一步获得一次奖励',angle:'越偏离竖直方向，惩罚越大',position:'越远离轨道中心，惩罚越大',failure:'倒杆或小车越界时扣分'});
const defaults = {alive:1,angle:.1,position:.05,failure:1};
const statuses = I18n.labels({starting:'正在初始化',running:'训练中',pausing:'等待暂停',paused:'已暂停',stopping:'正在保存',stopped:'已停止',completed:'已完成',failed:'训练失败',interrupted:'已中断'});
const active = ['starting','running','pausing','paused','stopping'];
let runs = [], run = null, selectedId = null, comparison = null, draft = true, mode = 'live';
let replay = null, replayKey = '', replayLoadingKey = '', frameIndex = 0, playing = false;
let pending = false, online = true, toastTimer, generation = 0, tickBusy = false;
let nf = new Intl.NumberFormat(I18n.locale);
const fmt = (n, digits=1) => Number.isFinite(n) ? (Math.abs(n)<0.5*10**(-digits)?0:n).toLocaleString(I18n.locale,{maximumFractionDigits:digits,minimumFractionDigits:digits}) : '—';
const esc = s => String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));

async function api(path, data) {
  const res = await fetch(path, data === undefined ? {} : {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});
  const body = await res.json();
  if (!res.ok) throw new Error(body.error || I18n.msg`请求失败 (${res.status})`);
  return body;
}
function toast(message) { $('toast').textContent=I18n.t(message); $('toast').hidden=false; clearTimeout(toastTimer); toastTimer=setTimeout(()=>$('toast').hidden=true,5000); }
function setOptions(el, options) { const html=options.map(([value,label])=>`<option value="${esc(value)}">${esc(label)}</option>`).join(''); if(el.innerHTML!==html){const old=el.value;el.innerHTML=html;if(options.some(o=>String(o[0])===old))el.value=old;} }
function readWeights() { return Object.fromEntries(Object.keys(defaults).map(k=>[k,Number($(`w-${k}`).value)])); }
function setWeights(weights) { for (const [k,v] of Object.entries(weights)) { $(`w-${k}`).value=v; $(`range-${k}`).value=v; } renderFrame(); }
function fillConfig(config) {
  $('name').value=config.name; $('seed').value=config.seed;
  for(const [id,v] of [['total-steps',config.total_steps],['learning-rate',config.learning_rate]]){
    const el=$(id); if(!Array.from(el.options).some(o=>o.value===String(v)))el.add(new Option(String(v),String(v)));el.value=v;
  }
  setWeights(config.weights);
}
for (const k of Object.keys(defaults)) {
  const el=document.createElement('div');el.className='weight-field';
  el.innerHTML=I18n.msg`<div class="weight-top"><label for="w-${k}">${labels[k]}</label><input id="w-${k}" type="number" min="0" max="10" step="0.01" required value="${defaults[k]}" aria-label="${labels[k]}权重"></div><span class="weight-hint">${hints[k]}</span><input id="range-${k}" type="range" min="0" max="10" step="0.01" value="${defaults[k]}" aria-label="${labels[k]}滑块">`;
  $('weight-fields').append(el);
  $(`w-${k}`).addEventListener('input',()=>{$(`range-${k}`).value=$(`w-${k}`).value;renderFrame();});
  $(`range-${k}`).addEventListener('input',()=>{$(`w-${k}`).value=$(`range-${k}`).value;renderFrame();});
  const bar=document.createElement('div');bar.className='reward-line';bar.innerHTML=`<div><span>${labels[k]}</span><b id="value-${k}">—</b></div><div class="reward-track ${k==='alive'?'':'negative'}"><i id="bar-${k}"></i></div>`;$('reward-bars').append(bar);
}

function renderList() {
  $('run-count').textContent=runs.length;
  const html=runs.length ? runs.map(r=>I18n.msg`<button class="run-item ${r.id===selectedId?'active':''}" data-id="${r.id}" title="${esc(r.config.name)}"><strong>${esc(r.config.name)}</strong><small><span>${statuses[r.status]||r.status}</span><span>${nf.format(r.step)} 步</span></small></button>`).join('') : I18n.t('<p class="muted">还没有实验，开始第一次训练吧。</p>');
  if($('run-list').innerHTML!==html)$('run-list').innerHTML=html;
  setOptions($('compare'),[['',I18n.t('不对比')],...runs.filter(r=>r.id!==selectedId).map(r=>[r.id,r.config.name])]);
}
function render() {
  renderList();
  const running=runs.find(r=>active.includes(r.status));
  const isActive=run && active.includes(run.status);
  $('run-title').textContent=run ? run.config.name : I18n.t('从一次实验开始');
  $('run-subtitle').textContent=run ? I18n.msg`${run.config.algorithm} · 随机种子 ${run.config.seed} · ${new Date(run.created_at).toLocaleString(I18n.locale)}` : I18n.t('调整奖励，看清策略如何学习。');
  $('status').textContent=run ? statuses[run.status] : I18n.t('准备就绪');$('status').className=`status ${run?.status||''}`;
  $('config-heading').textContent=draft ? (run?I18n.t('新实验草稿'):I18n.t('新实验配置')) : I18n.t('本次实验配置');
  $('config-fields').disabled=!draft;
  $('copy-config').hidden=!run;
  $('start').hidden=!draft;$('start').disabled=!!running||pending||!online;
  $('start').textContent=running ? I18n.t('已有实验运行中') : pending ? I18n.t('正在启动…') : I18n.t('▶ 开始训练');
  $('run-controls').hidden=!isActive;
  $('pause').disabled=!online||pending||!['running','starting','paused','pausing'].includes(run?.status);
  $('pause').textContent=['paused','pausing'].includes(run?.status)?I18n.t('继续训练'):I18n.t('暂停训练');
  $('stop').disabled=!online||pending||run?.status==='stopping';
  $('config-note').textContent=draft&&run ? I18n.t('已复制此实验的配置。可先在回放中试算新奖励，再开始独立训练。') : I18n.t('配置在实验开始后固定。修改奖励时，复制为新实验。');
  $('metric-step').textContent=run ? nf.format(run.step) : '—';
  $('metric-budget').textContent=run ? I18n.msg`/ ${nf.format(run.config.total_steps)} 步 · ${Math.round(run.step/run.config.total_steps*100)}%` : I18n.t('等待开始');
  $('progress-bar').style.width=run ? `${Math.min(100,run.step/run.config.total_steps*100)}%` : '0';
  const recent=run?.episodes.slice(-20)||[];
  $('metric-reward').textContent=recent.length ? fmt(recent.reduce((a,e)=>a+e.reward,0)/recent.length) : '—';
  const ev=run?.evaluations.at(-1);
  $('metric-eval').textContent=ev ? `${fmt(ev.mean_length)} / 500` : '—';
  $('metric-fps').textContent=run?.step ? I18n.msg`${nf.format(run.fps)} 步/s` : '—';
  $('metric-time').textContent=run ? I18n.msg`已用 ${fmt(run.elapsed,0)} 秒 · 含评估与暂停` : I18n.t('含评估与暂停的平均值');
  $('error-banner').hidden=!run?.error;$('error-banner').textContent=I18n.t(run?.error||'');
  const update=run?.updates.at(-1);
  $('signal-kl').textContent=fmt(update?.approx_kl,4);$('signal-clip').textContent=fmt(update?.clip_fraction,3);$('signal-variance').textContent=fmt(update?.explained_variance,3);
  $('downloads').hidden=!run;
  if(run){$('export').href=`/api/runs/${run.id}/export`;$('model-download').href=`/api/runs/${run.id}/model`;$('best-download').href=`/api/runs/${run.id}/best-model`;}
  $('model-download').hidden=!run?.model_saved;$('best-download').hidden=!run?.best_model_saved;
  const old=$('eval-select').value;
  setOptions($('eval-select'),(run?.evaluations||[]).map(e=>[e.index,I18n.msg`${e.step===0?I18n.t('训练前'):nf.format(e.step)+I18n.t(' 步')} · 均值 ${fmt(e.mean_length)}`]));
  if(old===''&&run?.evaluations.length)$('eval-select').value=run.evaluations.at(-1).index;
  $('view-replay').disabled=!run?.evaluations.length;
  $('replay-controls').hidden=mode!=='replay';
  $('view-live').classList.toggle('selected',mode==='live');$('view-replay').classList.toggle('selected',mode==='replay');
  $('scene-caption').textContent=mode==='live'?I18n.t('训练采样 · 每秒刷新 · 动作后的状态'):I18n.t('固定种子评估 · 确定性动作 · 动作后的状态');
  $('preview-panel').hidden=!(mode==='replay'&&draft&&run);
  if(mode==='replay')loadReplay();
  renderFrame();renderChart();
}
async function selectRun(id) {
  generation++;selectedId=id;draft=false;mode='live';playing=false;replay=null;replayKey='';comparison=null;$('compare').value='';
  const token=generation;
  try{const data=await api(`/api/runs/${id}`);if(token!==generation)return;run=data;fillConfig(data.config);render();}
  catch(e){toast(e.message);}
}
function newRun() {
  generation++;selectedId=null;run=null;draft=true;mode='live';replay=null;replayKey='';playing=false;comparison=null;$('compare').value='';
  fillConfig({name:I18n.msg`平衡练习 ${String(runs.length+1).padStart(2,'0')}`,seed:42,total_steps:20480,learning_rate:.0003,weights:defaults});render();$('name').focus();
}
$('run-list').addEventListener('click',e=>{const button=e.target.closest('[data-id]');if(button)selectRun(button.dataset.id);});
$('new-run').addEventListener('click',newRun);
$('copy-config').addEventListener('click',()=>{if(!run)return;draft=true;fillConfig({...run.config,name:run.config.name+I18n.t(' · 调整')});render();$('name').focus();toast(I18n.t('配置已复制，修改后会创建独立实验'));});
$('reset-weights').addEventListener('click',()=>setWeights(defaults));
$('config-form').addEventListener('submit',async e=>{
  e.preventDefault();if(pending||!draft)return;pending=true;render();
  try{const data=await api('/api/runs',{name:$('name').value,total_steps:Number($('total-steps').value),seed:Number($('seed').value),learning_rate:Number($('learning-rate').value),weights:readWeights()});
    runs=await api('/api/runs');await selectRun(data.id);toast(I18n.t('训练已启动，数据会自动保存'));
  }catch(e){toast(e.message);}finally{pending=false;render();}
});
async function control(action){if(!run||pending)return;pending=true;render();const id=run.id;try{const data=await api(`/api/runs/${id}/${action}`,{});if(selectedId===id)run=data;await tick();}catch(e){toast(e.message);}finally{pending=false;render();}}
$('pause').addEventListener('click',()=>control(['paused','pausing'].includes(run?.status)?'resume':'pause'));
$('stop').addEventListener('click',()=>control('stop'));
$('view-live').addEventListener('click',()=>{mode='live';playing=false;render();});
$('view-replay').addEventListener('click',()=>{mode='replay';if(!replayKey&&run?.evaluations.length)$('eval-select').value=run.evaluations.at(-1).index;render();});
$('eval-select').addEventListener('change',()=>{playing=false;loadReplay();});
$('episode-select').addEventListener('change',()=>{frameIndex=0;playing=false;renderFrame();});
$('frame-slider').addEventListener('input',()=>{frameIndex=Number($('frame-slider').value);playing=false;renderFrame();});
$('play').addEventListener('click',()=>{const frames=currentFrames();if(!frames.length)return;if(frameIndex>=frames.length-1)frameIndex=0;playing=!playing;renderFrame();});
$('chart-mode').addEventListener('change',renderChart);
$('compare').addEventListener('change',async()=>{const id=$('compare').value;comparison=null;if(id){try{const data=await api(`/api/runs/${id}`);if($('compare').value===id){comparison=data;$('chart-mode').value='evaluation';}}catch(e){toast(e.message);}}renderChart();});

async function loadReplay(){
  if(!run||!$('eval-select').value)return;
  const key=`${run.id}/eval/${$('eval-select').value}`;
  if(key===replayKey||key===replayLoadingKey)return;
  replayLoadingKey=key;replay=null;renderFrame();
  try{const data=await api(`/api/runs/${key}`);if(`${selectedId}/eval/${$('eval-select').value}`!==key)return;replay=data;replayKey=key;frameIndex=0;renderFrame();}
  catch(e){toast(e.message);}finally{if(replayLoadingKey===key)replayLoadingKey='';}
}
function currentFrames(){return replay?.episodes[Number($('episode-select').value)]?.frames||[];}
function renderFrame(){
  const frames=currentFrames();frameIndex=Math.min(frameIndex,Math.max(0,frames.length-1));
  const f=mode==='live'?run?.latest:frames[frameIndex];
  $('scene-empty').hidden=!!f;
  $('scene-empty').firstChild.textContent=mode==='replay'?I18n.t('正在载入评估轨迹'):run?I18n.t('等待训练状态'):I18n.t('让小车学会保持平衡');
  $('scene-empty').querySelector('small').textContent=run?I18n.t('初始化和评估期间，训练采样画面暂不更新'):I18n.t('点击右侧「开始训练」，运行真实 PPO 实验');
  $('scene-label').textContent=mode==='live' ? (run ? I18n.t('训练状态采样') : I18n.t('等待训练')) : I18n.msg`评估回放 · 种子 ${replay?.episodes[Number($('episode-select').value)]?.seed??'—'}`;
  $('scene-step').textContent=f ? (f.terminated?I18n.t('本回合失败'):f.truncated?I18n.t('达到 500 步上限'):mode==='live'?I18n.msg`训练第 ${nf.format(run.step)} 步`:I18n.msg`第 ${frameIndex+1} 步`) : I18n.t('CartPole · 状态示意');
  $('state-x').textContent=f?fmt(f.after[0],3)+' m':'—';$('state-v').textContent=f?fmt(f.after[1],3)+' m/s':'—';
  $('state-angle').textContent=f?fmt(f.after[2]*180/Math.PI,2)+'°':'—';$('state-action').textContent=f?(f.action===0?I18n.t('← 向左'):I18n.t('向右 →')):'—';$('state-reward').textContent=fmt(f?.reward,3);
  $('reward-total').textContent=fmt(f?.reward,4);$('reward-source').textContent=f?(mode==='live'?I18n.t('训练采样'):I18n.t('回放步骤')):I18n.t('等待数据');
  const magnitude=f?Math.max(...Object.values(f.components).map(Math.abs),.001):1;
  for(const k of Object.keys(defaults)){$(`value-${k}`).textContent=f?(f.components[k]>=0?'+':'')+fmt(f.components[k],4):'—';$(`bar-${k}`).style.width=f?`${Math.abs(f.components[k])/magnitude*100}%`:'0';}
  $('reward-footnote').textContent=f?I18n.msg`原始环境奖励 ${fmt(f.raw_reward,0)}；这里显示加权后的实际奖励，无归一化。`:I18n.t('与上方画面同一步，正负贡献分别显示。');
  $('frame-slider').max=Math.max(0,frames.length-1);$('frame-slider').value=frameIndex;$('frame-label').textContent=frames.length?`${frameIndex+1} / ${frames.length}`:'0 / 0';
  $('play').textContent=playing?I18n.t('暂停回放'):I18n.t('播放');$('play').disabled=!frames.length;
  if(frames.length){$('original-return').textContent=fmt(frames.reduce((s,f)=>s+f.reward,0));const w=readWeights();$('preview-return').textContent=fmt(frames.reduce((s,f)=>s+Object.keys(w).reduce((t,k)=>t+w[k]*f.features[k],0),0));}
  drawScene(f);
}
function canvasContext(id){const c=$(id),r=c.getBoundingClientRect(),d=window.devicePixelRatio||1;if(c.width!==Math.round(r.width*d)||c.height!==Math.round(r.height*d)){c.width=Math.round(r.width*d);c.height=Math.round(r.height*d);}const ctx=c.getContext('2d');ctx.setTransform(d,0,0,d,0,0);ctx.clearRect(0,0,r.width,r.height);return [ctx,r.width,r.height];}
function drawScene(f){
  const [ctx,w,h]=canvasContext('scene');const base=h*.76,scale=w/6.6;
  ctx.strokeStyle='#dfe9e2';ctx.lineWidth=1;
  for(let x=20;x<w;x+=30){ctx.beginPath();ctx.moveTo(x,45);ctx.lineTo(x,h-25);ctx.stroke();}
  for(let y=58;y<h-20;y+=30){ctx.beginPath();ctx.moveTo(15,y);ctx.lineTo(w-15,y);ctx.stroke();}
  ctx.strokeStyle='#91aa9b';ctx.beginPath();ctx.moveTo(20,base+17);ctx.lineTo(w-20,base+17);ctx.stroke();
  ctx.font='11px system-ui';ctx.fillStyle='#7d9385';ctx.textAlign='center';
  for(const v of [-2.4,0,2.4]){const x=w/2+v*scale;ctx.fillText(`${v} m`,x,base+38);ctx.beginPath();ctx.moveTo(x,base+12);ctx.lineTo(x,base+22);ctx.stroke();}
  if(!f)return;
  const x=w/2+f.after[0]*scale,angle=f.after[2],pole=Math.min(h*.44,120),cy=base-13;
  ctx.fillStyle='#d4e5da';ctx.beginPath();ctx.ellipse(x,base+18,44,5,0,0,Math.PI*2);ctx.fill();
  ctx.fillStyle='#296f50';ctx.beginPath();ctx.roundRect(x-30,base-14,60,25,5);ctx.fill();
  for(const dx of [-19,19]){ctx.fillStyle='#324b3b';ctx.beginPath();ctx.arc(x+dx,base+12,6,0,Math.PI*2);ctx.fill();ctx.fillStyle='#bfd3c6';ctx.beginPath();ctx.arc(x+dx,base+12,2,0,Math.PI*2);ctx.fill();}
  const tipX=x+Math.sin(angle)*pole,tipY=cy-Math.cos(angle)*pole;
  ctx.strokeStyle=f.terminated?'#c96b4b':'#dcac63';ctx.lineWidth=9;ctx.lineCap='round';ctx.beginPath();ctx.moveTo(x,cy);ctx.lineTo(tipX,tipY);ctx.stroke();ctx.lineWidth=1;
  ctx.fillStyle='#fff';ctx.beginPath();ctx.arc(x,cy,4,0,Math.PI*2);ctx.fill();ctx.strokeStyle='#336b4f';ctx.stroke();
  const dir=f.action===0?-1:1;ctx.strokeStyle='#36815e';ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(x+dir*40,base-2);ctx.lineTo(x+dir*65,base-2);ctx.lineTo(x+dir*59,base-7);ctx.moveTo(x+dir*65,base-2);ctx.lineTo(x+dir*59,base+3);ctx.stroke();
}
function renderChart(){
  const evaluation=$('chart-mode').value==='evaluation';
  $('chart-description').textContent=evaluation?I18n.t('固定评估种子的平均存活步数 · 每 2,048 步及训练结束评估'):I18n.t('每个训练回合的奖励与最近 20 回合均值');
  $('chart-legend').innerHTML=I18n.msg`<i></i>当前实验${comparison?I18n.t('<i class="compare-line"></i>对比实验'):''}`;
  const a=(evaluation?run?.evaluations:run?.episodes)||[],b=(evaluation?comparison?.evaluations:comparison?.episodes)||[];
  const [ctx,w,h]=canvasContext('chart');const pad={l:46,r:14,t:15,b:29},iw=w-pad.l-pad.r,ih=h-pad.t-pad.b;
  const points=data=>data.map((p,i)=>({x:p.step,y:evaluation?p.mean_length:data.slice(Math.max(0,i-19),i+1).reduce((s,e)=>s+e.reward,0)/Math.min(i+1,20)}));
  const ap=points(a),bp=points(b);const raw=a.map(p=>({x:p.step,y:evaluation?p.mean_length:p.reward}));const all=[...ap,...bp,...raw];
  let minY=evaluation?0:Math.min(0,...all.map(p=>p.y)),maxY=evaluation?500:Math.max(1,...all.map(p=>p.y));if(!evaluation){const range=maxY-minY;maxY+=range*.12;if(minY<0)minY-=range*.08;}
  const maxX=Math.max(1,run?.step||0,comparison?.step||0,...all.map(p=>p.x));
  const x=v=>pad.l+v/maxX*iw,y=v=>pad.t+ih-(v-minY)/(maxY-minY)*ih;
  ctx.font='11px system-ui';ctx.lineWidth=1;ctx.textAlign='right';
  for(let i=0;i<=4;i++){const v=minY+(maxY-minY)*i/4,py=y(v);ctx.strokeStyle='#edf1ee';ctx.beginPath();ctx.moveTo(pad.l,py);ctx.lineTo(w-pad.r,py);ctx.stroke();ctx.fillStyle='#87978d';ctx.fillText(fmt(v,0),pad.l-9,py+4);}
  ctx.textAlign='center';for(let i=0;i<=4;i++){const v=maxX*i/4;ctx.fillStyle='#87978d';ctx.fillText(v>=1000?fmt(v/1000,1)+'k':fmt(v,0),x(v),h-8);}
  function line(pts,color,width){if(!pts.length)return;ctx.strokeStyle=color;ctx.lineWidth=width;ctx.beginPath();pts.forEach((p,i)=>i?ctx.lineTo(x(p.x),y(p.y)):ctx.moveTo(x(p.x),y(p.y)));ctx.stroke();if(evaluation||pts.length===1){ctx.fillStyle=color;pts.forEach(p=>{ctx.beginPath();ctx.arc(x(p.x),y(p.y),3,0,Math.PI*2);ctx.fill();});}}
  if(!evaluation)line(raw,'#c4ddd0',1);
  line(ap,'#237753',2);line(bp,'#d7833b',2);
  if(!all.length){ctx.fillStyle='#87988e';ctx.font='13px system-ui';ctx.textAlign='center';ctx.fillText(I18n.t('训练开始后，曲线会出现在这里'),w/2,h/2);}
}
async function tick(){
  if(tickBusy)return;tickBusy=true;const token=generation;
  try{const data=await api('/api/runs');runs=data;online=true;$('connection').textContent=I18n.t('服务已连接 · 每秒更新');
    if(selectedId){const current=await api(`/api/runs/${selectedId}`);if(token===generation)run=current;}
    const cmpId=$('compare').value;if(cmpId&&runs.find(r=>r.id===cmpId&&active.includes(r.status))){const c=await api(`/api/runs/${cmpId}`);if($('compare').value===cmpId)comparison=c;}
    if(token===generation)render();
  }catch(e){online=false;$('connection').textContent=I18n.t('连接断开 · 正在重试');$('start').disabled=true;$('pause').disabled=true;$('stop').disabled=true;}
  finally{tickBusy=false;}
}
setInterval(tick,1000);
setInterval(()=>{if(!playing||mode!=='replay')return;const frames=currentFrames();if(frameIndex<frames.length-1)frameIndex++;else playing=false;renderFrame();},40);
new ResizeObserver(()=>{renderFrame();renderChart();}).observe(document.querySelector('.main-column'));
async function init(){await tick();if(runs.length)await selectRun(runs.find(r=>active.includes(r.status))?.id||runs[0].id);else render();}
init();
window.addEventListener('languagechange',()=>{
 $('connection').textContent=I18n.t(online?'服务已连接 · 每秒更新':'连接断开 · 正在重试');
 nf=new Intl.NumberFormat(I18n.locale);
 for(const k of Object.keys(defaults)){
  const input=$(`w-${k}`),row=input.closest('.weight-field');
  row.querySelector('label').textContent=labels[k];
  row.querySelector('.weight-hint').textContent=hints[k];
  input.setAttribute('aria-label',labels[k]+I18n.t('权重'));
  $(`range-${k}`).setAttribute('aria-label',labels[k]+I18n.t('滑块'));
  $(`value-${k}`).parentElement.querySelector('span').textContent=labels[k];
 }
 render();
});
