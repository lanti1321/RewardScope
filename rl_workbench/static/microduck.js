'use strict';
const $=id=>document.getElementById(id);
const escapeHTML=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const format=(n,d=2)=>Number.isFinite(n)?(Math.abs(n)<.5*10**(-d)?0:n).toLocaleString(I18n.locale,{maximumFractionDigits:d}):'—';
const statusNames=I18n.labels({starting:'初始化中',running:'训练中',paused:'已暂停',stopped:'已停止',completed:'已完成',failed:'失败',interrupted:'已中断'});
let connectionState='connecting';
let catalog={ready:false,tasks:[]},runs=[],run=null,selected=null,draft=true,weights={},disabledRewards=new Set(),frameIndex=0,playing=false,busy=false,polling=false,revision=0,toastTimer;
const isActive=r=>r&&['starting','running','paused','pausing','stopping'].includes(r.status);
async function api(path,data){const response=await fetch(path,data===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});const result=await response.json();if(!response.ok)throw new Error(result.error||I18n.t('请求失败'));return result;}
function toast(text){$('toast').textContent=I18n.t(text);$('toast').hidden=false;clearTimeout(toastTimer);toastTimer=setTimeout(()=>$('toast').hidden=true,5500);}
function options(el,items){const html=items.map(([value,label])=>`<option value="${escapeHTML(value)}">${escapeHTML(label)}</option>`).join('');if(el.innerHTML!==html){const previous=el.value;el.innerHTML=html;if(items.some(([v])=>String(v)===previous))el.value=previous;}}
function task(){return catalog.tasks.find(t=>t.id===$('duck-task').value);}
// Scrolling a long configuration form must never change focused numeric fields.
$('duck-form').addEventListener('wheel',e=>{if(e.target.matches('input[type="number"]'))e.target.blur();},{capture:true,passive:true});
function renderStartLabel(){const iterations=Number($('duck-iters').value);$('duck-start').textContent=runs.some(isActive)?I18n.t('已有训练运行中'):I18n.msg`▶ 启动训练 · ${Number.isInteger(iterations)&&iterations>0?format(iterations,0):'—'} 轮`;}
$('duck-iters').addEventListener('input',renderStartLabel);
function setView(view){document.querySelector('main').dataset.view=view;window.scrollTo({top:0,behavior:'instant'});document.querySelectorAll('.workspace-nav button').forEach(b=>{if(b.dataset.view===view)b.setAttribute('aria-current','page');else b.removeAttribute('aria-current');});requestAnimationFrame(()=>{drawChart();renderEvidence();});}
document.querySelectorAll('.workspace-nav button').forEach(b=>b.addEventListener('click',()=>setView(b.dataset.view)));
$('project-configure').addEventListener('click',()=>setView('config'));
function drawWeights(){
 const t=task();for(const role of ['actor','critic']){$('network-'+role).value=(t?.network?.[role]?.hidden_dims||[]).join(', ');$('network-'+role).disabled=t?.network?.[role]?.class_name!=='MLPModel';}
 $('duck-weight-fields').innerHTML=(t?.rewards||[]).map(r=>I18n.msg`<div class="duck-weight"><div class="duck-weight-top"><label for="weight-${escapeHTML(r.name)}">${escapeHTML(r.name)}</label><input type="number" min="-1000000" max="1000000" step="any" value="${weights[r.name]??r.weight}" id="weight-${escapeHTML(r.name)}" data-term="${escapeHTML(r.name)}" aria-label="${escapeHTML(r.name)} 权重" ${disabledRewards.has(r.name)?'disabled':''} required></div><div class="reward-options"><label><input type="checkbox" data-disable="${escapeHTML(r.name)}" aria-label="屏蔽 ${escapeHTML(r.name)}" ${disabledRewards.has(r.name)?'checked':''}>屏蔽</label><small id="weight-mode-${escapeHTML(r.name)}">${Object.hasOwn(weights,r.name)?I18n.t('固定权重'):I18n.t('跟随官方课程')} · 官方初始值 ${format(r.weight,5)}</small><button type="button" class="text-button" data-restore="${escapeHTML(r.name)}">恢复</button></div></div>`).join('');
 renderTaskInfo();
}
function renderTaskInfo(){const t=task();
 $('duck-task-info').textContent=t?I18n.msg`${t.rewards.length} 个奖励项 · ${t.curricula.length} 个课程项 · 控制频率 ${format(1/t.step_dt,0)} Hz。每次迭代采集 ${t.steps_per_iteration} 步 / 环境。`:'';
}
function renderEnvironment(){const t=task();$('environment-task').textContent=t?.id||I18n.t('等待项目配置');$('environment-facts').innerHTML=t?[[I18n.t('控制频率'),`${format(1/t.step_dt,0)} Hz`],[I18n.t('回合时长上限'),`${format(t.episode_seconds)} s`],[I18n.t('每次更新采样'),I18n.msg`${t.steps_per_iteration} 步 / 环境`],[I18n.t('奖励时间缩放'),t.scale_by_dt?I18n.t('按 dt 缩放'):I18n.t('不按 dt 缩放')]].map(([k,v])=>`<div><small>${k}</small><strong>${v}</strong></div>`).join(''):'';$('environment-contract').innerHTML=t?[[I18n.t('观测组'),t.observation_groups],[I18n.t('终止条件'),t.terminations],[I18n.t('课程项'),t.curricula]].map(([label,items])=>`<div class="environment-group"><b>${label}</b><div>${items.map(x=>`<code>${escapeHTML(x)}</code>`).join('')||I18n.t('无')}</div></div>`).join(''):'';}
function fill(config){$('duck-name').value=config.name;$('duck-task').value=config.task;$('duck-envs').value=config.num_envs;$('duck-iters').value=config.iterations;$('duck-seed').value=config.seed;weights={...config.reward_weights};disabledRewards=new Set(config.disabled_rewards||[]);for(const [name,scale] of Object.entries(config.reward_scales||{})){if(scale===0)disabledRewards.add(name);else if(scale!==1&&!Object.hasOwn(weights,name)){const r=task()?.rewards.find(r=>r.name===name);if(r)weights[name]=r.weight*scale;}}drawWeights();for(const [role,dims] of Object.entries(config.network||{}))$('network-'+role).value=dims.join(', ');}
function render(){
 renderEnvironment();
 $('duck-count').textContent=runs.length;
 const list=runs.map(r=>I18n.msg`<div class="run-card"><button type="button" class="run-item ${r.id===selected?'active':''}" data-run="${r.id}"><strong>${escapeHTML(r.config.name)}</strong><small><span>${statusNames[r.status]||r.status}</span><span>${r.iteration} 次更新</span></small></button><button type="button" class="run-delete" data-delete-run="${r.id}" ${busy||isActive(r)?'disabled':''} title="${escapeHTML(isActive(r)?I18n.t('请先停止训练并等待保存完成'):I18n.t('删除训练记录'))}" aria-label="${escapeHTML(I18n.t('删除训练记录')+' · '+r.config.name+' · '+r.id)}"><svg viewBox="0 0 24 24" width="17" height="17" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M3 6h18M9 6V3h6v3M5 6l1 15h12l1-15M10 10v7M14 10v7"/></svg></button></div>`).join('')||I18n.t('<p class="muted">暂无训练记录。</p>');
 if($('duck-runs').innerHTML!==list)$('duck-runs').innerHTML=list;
 $('duck-title').textContent=run?run.config.name:'RL Workbench';
 $('duck-subtitle').textContent=run?I18n.msg`${run.config.task} · ${run.config.num_envs} 个环境 · 种子 ${run.config.seed}`:I18n.t('MicroDuck 训练工作台');
 $('duck-status').textContent=run?statusNames[run.status]:I18n.t('准备就绪');$('duck-status').className=`status ${run?.status||''}`;
 $('duck-error').hidden=!(run?.error||run?.render_error||!catalog.ready);$('duck-error').textContent=I18n.t(run?.error||run?.render_error||catalog.error||'');
 const source=run?.source||catalog.source;$('duck-commit').textContent=source?source.commit.slice(0,10)+(source.dirty?I18n.t(' · 有改动'):I18n.t(' · 干净版本')):I18n.t('未安装');
 $('duck-gpu').textContent=run?.gpu||catalog.gpu||I18n.t('需要 CUDA GPU');
 const metrics=run?.metrics||{};
 $('duck-iteration').textContent=run?`${run.iteration} / ${run.config.iterations}`:'—';$('duck-phase').textContent=I18n.t(run?.phase||'首次 CUDA 编译可能需要几分钟');
 $('duck-progress').style.width=run?`${Math.min(100,run.iteration/run.config.iterations*100)}%`:'0';
 $('duck-reward').textContent=format(metrics['Train/mean_reward']);$('duck-length').textContent=format(metrics['Train/mean_episode_length'],1);$('duck-steps').textContent=run?format(run.step,0):'—';
 $('duck-speed').textContent=run?I18n.msg`${format(metrics['Perf/total_fps'],0)} 步/s · 已用 ${format(run.elapsed,0)} 秒`:I18n.t('并行环境数 × 每环境步数');
 $('duck-config-title').textContent=draft?(run?I18n.t('配置草稿'):I18n.t('训练配置')):I18n.t('本次训练配置');$('duck-copy').hidden=!run;$('duck-fields').disabled=!draft||!catalog.ready;
 $('duck-start').hidden=!draft;$('duck-start').disabled=busy||!catalog.ready||runs.some(isActive);renderStartLabel();
 $('duck-controls').hidden=!isActive(run);$('duck-pause').disabled=busy;$('duck-stop').disabled=busy;$('duck-pause').textContent=run?.status==='paused'?I18n.t('继续训练'):I18n.t('暂停训练');
 const downloads=run?I18n.msg`<a class="download" href="/api/microduck/runs/${run.id}/export">导出训练记录 JSON（含采样数据）</a>`+(run.checkpoints||[]).slice(-3).map(n=>I18n.msg`<a class="download" href="/api/microduck/runs/${run.id}/download/${n}">下载 ${escapeHTML(n)}</a>`).join('')+(run.onnx||[]).map(n=>I18n.msg`<a class="download" href="/api/microduck/runs/${run.id}/download/${n}">下载官方导出 ONNX</a>`).join(''):'';
 if($('duck-downloads').innerHTML!==downloads)$('duck-downloads').innerHTML=downloads;
 const contract=run?.observation_dims?I18n.msg`观测：${Object.entries(run.observation_dims).map(([k,v])=>escapeHTML(k)+' '+v.join('×')).join(' / ')}<br>动作维度：${run.action_dim}<br>控制步长：${format(run.step_dt,3)} s`:I18n.t('环境创建后显示真实观测与动作维度。');$('duck-contract').innerHTML=contract;
 $('duck-signals').innerHTML=[[I18n.t('学习率'),'Loss/learning_rate'],[I18n.t('动作标准差'),'Policy/mean_std'],[I18n.t('采样耗时'),'Perf/collection_time'],[I18n.t('优化耗时'),'Perf/learning_time']].map(([label,key])=>`<div><span>${label}</span><b>${format(metrics[key],5)}</b></div>`).join('');
 if($('duck-follow').checked&&run?.frames.length)frameIndex=run.frames.length-1;
 renderFrame();drawChart();renderOverallReward();
}
function renderFrame(){
 const frames=run?.frames||[];frameIndex=Math.max(0,Math.min(frameIndex,frames.length-1));const f=frames[frameIndex];
 if(!run?.live_preview){
 $('duck-frame').hidden=!f?.image;$('duck-empty').hidden=!!f?.image;
 $('duck-empty').querySelector('strong').textContent=run?I18n.t(run.phase):I18n.t('等待训练环境画面');
 if(f?.image){const url=`/api/microduck/runs/${run.id}/frame/${f.index}`;if($('duck-frame').getAttribute('src')!==url)$('duck-frame').src=url;}
 $('duck-frame-tag').textContent=f?I18n.msg`历史采样 · 环境 0 · 第 ${f.env_step} 步`:'';
 }
 $('duck-scrub').max=Math.max(0,frames.length-1);$('duck-scrub').value=frameIndex;$('duck-frame-count').textContent=frames.length?`${frameIndex+1} / ${frames.length}`:'0 / 0';
 $('duck-play').disabled=!frames.length;$('duck-play').textContent=playing?I18n.t('暂停数据回放'):I18n.t('播放数据采样');
 $('duck-envstep').textContent=f?format(f.env_step,0):'—';$('duck-epstep').textContent=f?format(f.episode_step,0):'—';$('duck-step-reward').textContent=format(f?.reward,4);$('duck-done').textContent=f?(f.terminated?I18n.t('失败终止'):f.truncated?I18n.t('时间截断'):I18n.t('进行中')):'—';
 const episode=$('duck-reward-mode').value==='episode';
 const terms=episode?Object.fromEntries(Object.entries(run?.metrics||{}).filter(([k])=>k.startsWith('Episode_Reward/')).map(([k,v])=>[k.slice(15),v])):f?.components||{};
 const max=Math.max(.000001,...Object.values(terms).map(Math.abs));
 $('duck-reward-terms').innerHTML=Object.keys(terms).length?Object.entries(terms).sort((a,b)=>Math.abs(b[1])-Math.abs(a[1])).map(([key,value])=>`<div class="reward-line"><div><span>${escapeHTML(key)}</span><b>${value>0?'+':''}${format(value,5)}</b></div><div class="reward-track ${value<0?'negative':''}"><i style="width:${Math.abs(value)/max*100}%"></i></div></div>`).join(''):I18n.t('<p class="note">等待该口径的数据。官方回合日志需要环境产生结束回合后才有值。</p>');
 $('duck-reward-caption').textContent=episode?I18n.t('与官方 TensorBoard 相同的 Episode_Reward 指标'):I18n.t('环境 0 · 当前采样步的真实加权贡献');
 $('duck-reward-note').textContent=episode?I18n.t('官方日志将已结束回合的分项累计奖励除以回合时长上限（本任务通常为 20 秒），再按训练迭代聚合；不是回合总分。要分解顶部平均回合奖励，请看“整体奖励构成”。'):f?I18n.msg`包含 dt 缩放；分项求和与总奖励误差 ${format(f.sum_residual,9)}。未修改的权重跟随官方课程，固定权重和屏蔽以本次设置为准。`:I18n.t('单步贡献包含官方 dt 缩放；各项之和对应环境返回的奖励。');
 renderEvidence();
}
function drawChart(){
 const canvas=$('duck-chart'),rect=canvas.getBoundingClientRect(),d=window.devicePixelRatio||1;
 if(canvas.width!==Math.round(rect.width*d)||canvas.height!==Math.round(rect.height*d)){canvas.width=Math.round(rect.width*d);canvas.height=Math.round(rect.height*d);}
 if(!rect.width||!rect.height)return;
 const ctx=canvas.getContext('2d');ctx.setTransform(d,0,0,d,0,0);const w=rect.width,h=rect.height;ctx.clearRect(0,0,w,h);const key=$('duck-chart-metric').value;
 const points=r=>(r?.history||[]).filter(p=>Number.isFinite(p.metrics[key])).map(p=>({x:p.iteration,y:p.metrics[key]}));const a=points(run),all=a;
 const min=all.reduce((v,p)=>Math.min(v,p.y),0),max=all.reduce((v,p)=>Math.max(v,p.y),1)*1.1,xmax=Math.max(1,run?.iteration||0);
 const x=v=>48+v/xmax*(w-64),y=v=>15+(h-45)*(1-(v-min)/(max-min));
 ctx.font='11px system-ui';ctx.lineWidth=1;
 for(let i=0;i<=4;i++){const val=min+(max-min)*i/4;ctx.strokeStyle='#eaf0ec';ctx.beginPath();ctx.moveTo(48,y(val));ctx.lineTo(w-16,y(val));ctx.stroke();ctx.fillStyle='#7d9184';ctx.textAlign='right';ctx.fillText(format(val,Math.abs(max)<3?2:0),40,y(val)+4);ctx.textAlign='center';ctx.fillText(format(xmax*i/4,1),x(xmax*i/4),h-8);}
 for(const [pts,color] of [[a,'#237753']]){ctx.strokeStyle=color;ctx.fillStyle=color;ctx.lineWidth=2;ctx.beginPath();pts.forEach((p,i)=>i?ctx.lineTo(x(p.x),y(p.y)):ctx.moveTo(x(p.x),y(p.y)));ctx.stroke();if(pts.length<30)pts.forEach(p=>{ctx.beginPath();ctx.arc(x(p.x),y(p.y),3,0,Math.PI*2);ctx.fill();});}
 if(!all.length){ctx.textAlign='center';ctx.fillStyle='#83998a';ctx.font='13px system-ui';ctx.fillText(I18n.t('等待官方训练指标'),w/2,h/2);}
}
async function select(id){resetPreview();const token=++revision;selected=id;draft=false;playing=false;$('duck-follow').checked=true;try{const data=await api(`/api/microduck/runs/${id}`);if(token!==revision)return;run=data;fill(run.config);render();setView('monitor');}catch(e){toast(e.message);}}
function fresh(){resetPreview();revision++;selected=null;run=null;draft=true;playing=false;fill({name:I18n.t('我的 RL 训练'),task:catalog.default_task,num_envs:64,iterations:5,seed:42,reward_weights:{},disabled_rewards:[]});render();setView('config');$('duck-name').focus();}
$('duck-new').addEventListener('click',fresh);
$('duck-runs').addEventListener('click',e=>{const deletion=e.target.closest('[data-delete-run]');if(deletion){if(!deletion.disabled)deleteRun(deletion.dataset.deleteRun);return;}const b=e.target.closest('[data-run]');if(b)select(b.dataset.run);});
$('duck-task').addEventListener('change',()=>{weights={};disabledRewards=new Set();drawWeights();renderEnvironment();});
$('duck-weight-fields').addEventListener('input',e=>{const name=e.target.dataset.term;if(name){weights[name]=Number(e.target.value);const r=task().rewards.find(r=>r.name===name);$('weight-mode-'+name).textContent=I18n.msg`固定权重 · 官方初始值 ${format(r.weight,5)}`;}});
$('duck-weight-fields').addEventListener('change',e=>{const name=e.target.dataset.disable;if(name){if(e.target.checked)disabledRewards.add(name);else disabledRewards.delete(name);$('weight-'+name).disabled=e.target.checked;}});
$('duck-weight-fields').addEventListener('click',e=>{const name=e.target.dataset.restore;if(name){delete weights[name];disabledRewards.delete(name);const network=readNetwork();drawWeights();for(const [role,dims] of Object.entries(network))$('network-'+role).value=dims.join(', ');}});
$('duck-reset').addEventListener('click',()=>{const network=readNetwork();weights={};disabledRewards=new Set();drawWeights();for(const [role,dims] of Object.entries(network))$('network-'+role).value=dims.join(', ');renderEnvironment();});
$('duck-copy').addEventListener('click',()=>{draft=true;fill({...run.config,name:run.config.name+I18n.t(' · 调整')});render();setView('config');$('duck-name').focus();});
$('duck-form').addEventListener('submit',async e=>{e.preventDefault();if(busy)return;busy=true;render();try{const data=await api('/api/microduck/runs',{name:$('duck-name').value,task:$('duck-task').value,num_envs:Number($('duck-envs').value),iterations:Number($('duck-iters').value),seed:Number($('duck-seed').value),reward_weights:weights,disabled_rewards:[...disabledRewards],network:readNetwork()});runs=await api('/api/microduck/runs');await select(data.id);toast(I18n.t('官方训练已启动，首次编译需要一些时间'));}catch(e){toast(e.message);}finally{busy=false;render();}});
async function deleteRun(id){
 const target=runs.find(r=>r.id===id);
 if(!target||busy||isActive(target))return;
 const name=target.config.name;
 if(!confirm(I18n.msg`删除训练记录“${name}”（${id}）？\n将永久删除该记录的模型、日志、画面和统计数据，无法撤销。需要保留的模型请先下载。`))return;
 busy=true;render();
 try{
  await api(`/api/microduck/runs/${id}/delete`,{});
  if(selected===id)fresh();
  runs=await api('/api/microduck/runs');toast(I18n.t('训练记录及其文件已删除'));
 }catch(e){toast(e.message);}
 finally{busy=false;render();}
}
async function control(action){if(!run||busy)return;busy=true;render();try{await api(`/api/microduck/runs/${run.id}/${action}`,{});toast(action==='stop'?I18n.t('已请求停止，将在下一个采样边界保存模型'):action==='pause'?I18n.t('已请求暂停，将在下一个采样边界生效'):I18n.t('已请求继续训练'));}catch(e){toast(e.message);}finally{busy=false;render();}}
$('duck-pause').addEventListener('click',()=>control(run?.status==='paused'?'resume':'pause'));
$('duck-stop').addEventListener('click',()=>control('stop'));
$('duck-follow').addEventListener('change',()=>{playing=false;render();});
$('duck-scrub').addEventListener('input',()=>{$('duck-follow').checked=false;playing=false;frameIndex=Number($('duck-scrub').value);renderFrame();});
$('duck-play').addEventListener('click',()=>{$('duck-follow').checked=false;if(frameIndex>=(run?.frames.length||0)-1)frameIndex=0;playing=!playing;renderFrame();});
$('duck-chart-metric').addEventListener('change',drawChart);$('duck-reward-mode').addEventListener('change',renderFrame);
async function poll(){if(polling)return;polling=true;const token=revision;try{runs=await api('/api/microduck/runs');if(selected){const result=await api(`/api/microduck/runs/${selected}`);if(token===revision)run=result;}if($('duck-log-details').open&&selected){const logs=await api(`/api/microduck/runs/${selected}/log`);if(token===revision)$('duck-log').textContent=logs.text.replace(/\x1b\[[0-9;]*m/g,'');}connectionState='connected';$('duck-connection').textContent=I18n.t('已连接 · 数据保存在本机');if(token===revision)render();}catch(e){connectionState='disconnected';$('duck-connection').textContent=I18n.t('连接断开 · 重试中');$('duck-start').disabled=true;$('duck-pause').disabled=true;$('duck-stop').disabled=true;}finally{polling=false;}}
$('duck-log-details').addEventListener('toggle',()=>{if($('duck-log-details').open)poll();});
setInterval(poll,1200);setInterval(()=>{if(!playing)return;if(frameIndex<(run?.frames.length||0)-1)frameIndex++;else playing=false;renderFrame();},200);
new ResizeObserver(drawChart).observe($('duck-chart'));
async function init(){try{catalog=await api('/api/microduck/info');options($('duck-task'),catalog.tasks.map(t=>[t.id,t.id]));if(catalog.default_task)$('duck-task').value=catalog.default_task;drawWeights();renderDiscovery();await poll();fresh();setView('project');}catch(e){toast(e.message);}}
init();

function readNetwork(){const result={};for(const role of ['actor','critic']){const value=$('network-'+role).value.trim();if(value&&!$('network-'+role).disabled)result[role]=value.split(',').map(x=>Number(x.trim()));}return result;}
function renderDiscovery(){$('discovery-summary').textContent=I18n.msg`已读取 ${catalog.tasks.length} 个任务`; }
$('project-rescan').addEventListener('click',async()=>{const b=$('project-rescan');b.disabled=true;b.textContent=I18n.t('读取中…');try{catalog=await api('/api/microduck/scan',{});options($('duck-task'),catalog.tasks.map(t=>[t.id,t.id]));renderDiscovery();if(draft)drawWeights();render();toast(I18n.t('已重新读取项目代码与官方配置'));}catch(e){toast(e.message);}finally{b.disabled=false;b.textContent=I18n.t('刷新项目配置');}});
function diagnosticCanvas(id){const c=$(id),r=c.getBoundingClientRect();if(!r.width||!r.height)return null;const d=devicePixelRatio||1;c.width=Math.round(r.width*d);c.height=Math.round(r.height*d);const ctx=c.getContext('2d');ctx.setTransform(d,0,0,d,0,0);ctx.font='11px system-ui';return {ctx,w:r.width,h:r.height};}
function renderEvidence(){
 const f=run?.frames?.[frameIndex];const available=!!f?.reward_stats;$('evidence-empty').hidden=available;$('evidence-empty').textContent=isActive(run)?I18n.t('等待首个诊断采样点…'):I18n.t('此记录未采集内部诊断数据，请启动一次新训练。');$('evidence-body').hidden=!available;$('evidence-time').textContent=f?I18n.msg`环境步 ${f.env_step} · 环境 0 回合 ${f.episode_id??'—'}`:I18n.t('跟随上方采样时间轴');if(!available)return;
 options($('evidence-reward'),Object.keys(f.reward_stats).map(k=>[k,k]));const term=$('evidence-reward').value,stats=f.reward_stats[term];
 $('evidence-reward-stats').innerHTML=[[I18n.t('非零贡献频率'),format(stats.frequency*100,1)+'%'],[I18n.t('非零次数 / 总环境步'),`${stats.nonzero_count} / ${stats.samples}`],[I18n.t('实际计算环境步'),format(stats.evaluated_samples,0)],[I18n.t('启用时原始值均值'),format(stats.raw_mean_when_enabled,4)]].map(([k,v])=>`<div>${k}<b>${v}</b></div>`).join('');
 const chart=diagnosticCanvas('reward-trend'),frames=run.frames;
 if(chart){const {ctx,w,h}=chart,vals=frames.map(p=>p.reward_batch_mean?.[term]).filter(Number.isFinite),lo=Math.min(0,...vals),hi=Math.max(1e-7,...vals),x=i=>44+i/Math.max(1,frames.length-1)*(w-55),y=v=>15+(h-40)*(1-(v-lo)/(hi-lo));ctx.strokeStyle='#dce5df';ctx.beginPath();ctx.moveTo(44,y(0));ctx.lineTo(w-10,y(0));ctx.stroke();ctx.fillStyle='#63776a';ctx.fillText(format(hi,4),0,16);ctx.fillText(format(lo,4),0,h-26);ctx.fillText(String(frames[0].env_step),44,h-5);ctx.fillText(String(frames.at(-1).env_step),w-40,h-5);ctx.strokeStyle='#237753';ctx.beginPath();let begun=false;frames.forEach((p,i)=>{const v=p.reward_batch_mean?.[term];if(Number.isFinite(v)){begun?ctx.lineTo(x(i),y(v)):ctx.moveTo(x(i),y(v));begun=true;}});ctx.stroke();ctx.strokeStyle='#d7833b';ctx.beginPath();ctx.moveTo(x(frameIndex),10);ctx.lineTo(x(frameIndex),h-20);ctx.stroke();}

}
function chooseEvidenceFrame(i){$('duck-follow').checked=false;playing=false;frameIndex=i;renderFrame();}
$('reward-trend').addEventListener('click',e=>{const r=e.currentTarget.getBoundingClientRect();chooseEvidenceFrame(Math.max(0,Math.min(run.frames.length-1,Math.round((e.clientX-r.left-44)/(r.width-55)*(run.frames.length-1)))));});
$('evidence-reward').addEventListener('change',renderEvidence);
new ResizeObserver(renderEvidence).observe($('reward-trend'));

function renderOverallReward(){
 const b=run?.reward_breakdown;
 $('overall-reward-summary').innerHTML='';$('overall-reward-terms').innerHTML='';
 if(!b){$('overall-reward-caption').textContent=I18n.t('最近已结束回合');$('overall-reward-note').textContent=!run?I18n.t('启动训练后，显示与顶部平均回合奖励对应的分项贡献。'):run?.diagnostics_version>=2?I18n.t('等待回合结束后生成整体奖励分解。'):I18n.t('旧记录未采集同口径回合分项；新训练将自动记录，现有单步数据无法准确还原顶部总分。');return;}
 const total=b.logged_total??b.component_total,rows=Object.entries(b.components),residual=b.residual??0;
 if(Math.abs(residual)>1e-6)rows.push([I18n.t('日志差额（额外奖励或累计误差）'),residual]);
 const positive=rows.reduce((sum,[,v])=>sum+Math.max(0,v),0),negative=rows.reduce((sum,[,v])=>sum+Math.min(0,v),0),magnitude=positive-negative,percentOK=Math.abs(total)>Math.max(1e-8,magnitude*1e-4),max=Math.max(1e-9,...rows.map(([,v])=>Math.abs(v)));
 $('overall-reward-caption').textContent=I18n.msg`最近 ${b.episodes} 个已结束回合 · 最多 ${b.window} 个`;
 $('overall-reward-summary').innerHTML=[[I18n.t('总奖励'),total],[I18n.t('正贡献合计'),positive],[I18n.t('负贡献合计'),negative]].map(([label,value])=>`<div>${label}<b>${format(value,4)}</b></div>`).join('');
 $('overall-reward-terms').innerHTML=rows.sort((a,b)=>Math.abs(b[1])-Math.abs(a[1])).map(([name,value])=>`<div class="reward-line"><div><span>${escapeHTML(name)}</span><b>${value>0?'+':''}${format(value,4)}${percentOK?`（${format(value/total*100,1)}%）`:''}</b></div><div class="reward-track ${value<0?'negative':''}"><i style="width:${Math.abs(value)/max*100}%"></i></div></div>`).join('');
 $('overall-reward-note').textContent=I18n.t('每项为同一批已结束回合的平均累计贡献，正负相加对应顶部总奖励。占比 = 分项 ÷ 净总奖励，可能超过 100% 或为负；不是饼图份额。此面板跟随最新策略更新，不跟随采样回放。')+(percentOK?'':I18n.t('当前净奖励接近零，隐藏不稳定的百分比。'));
}

let previewBusy=false,previewRun=null,previewTime=null,previewStatus=null;
function resetPreview(){previewRun=null;previewTime=null;previewStatus=null;$('duck-frame').hidden=true;$('duck-frame').removeAttribute('src');$('duck-empty').hidden=false;$('duck-frame-tag').textContent='';}
async function pollPreview(){
 if(previewBusy||!run?.live_preview||document.hidden)return;
 const id=run.id;
 if(!isActive(run)&&previewRun===id&&previewTime!==null&&previewStatus===run.status)return;
 previewBusy=true;
 try{
  const frame=await api(`/api/microduck/runs/${id}/latest`);
  if(run?.id!==id)return;
  if(previewRun!==id||previewTime!==frame.captured_at){
   $('duck-frame').src=frame.image;previewRun=id;previewTime=frame.captured_at;
   $('duck-frame').hidden=false;$('duck-empty').hidden=true;
  }
  previewStatus=run.status;
  const age=Math.max(0,Math.floor(Date.now()/1000-frame.captured_at));
  $('duck-frame-tag').textContent=I18n.msg`${isActive(run)?I18n.t('最新画面'):I18n.t('最后画面')} · 环境 0 · 第 ${frame.env_step} 步 · ${age} 秒前`;
 }catch(e){if(run?.id===id)$('duck-frame-tag').textContent=I18n.t('等待最新画面');}
 finally{previewBusy=false;}
}
setInterval(pollPreview,500);
window.addEventListener('languagechange',()=>{
 $('duck-connection').textContent=I18n.t(connectionState==='connected'?'已连接 · 数据保存在本机':connectionState==='disconnected'?'连接断开 · 重试中':'连接本地训练服务…');
 // Relabel existing fields without resetting any draft values or focus.
 for(const r of task()?.rewards||[]){
  const input=$('weight-'+r.name);if(!input)continue;
  input.setAttribute('aria-label',I18n.msg`${r.name} 权重`);
  const row=input.closest('.duck-weight'),checkbox=row.querySelector('[data-disable]');
  checkbox.setAttribute('aria-label',I18n.msg`屏蔽 ${r.name}`);
  checkbox.parentElement.lastChild.textContent=I18n.t('屏蔽');
  row.querySelector('[data-restore]').textContent=I18n.t('恢复');
  $('weight-mode-'+r.name).textContent=(Object.hasOwn(weights,r.name)?I18n.t('固定权重'):I18n.t('跟随官方课程'))+I18n.msg` · 官方初始值 ${format(r.weight,5)}`;
 }
 renderTaskInfo();renderDiscovery();render();previewStatus=null;pollPreview();
});
