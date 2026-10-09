'use strict';
const $=id=>document.getElementById(id);
let lastReport=null,lastError=null;
$('scan-form').addEventListener('submit',async event=>{
 event.preventDefault();lastReport=null;lastError=null;$('scan-button').disabled=true;$('scan-status').textContent=I18n.t('正在扫描');$('scan-result').replaceChildren();$('scan-note').textContent='';
 try{
  const response=await fetch('/api/projects/scan',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({directory:$('project-root').value})});
  const report=await response.json();if(!response.ok)throw new Error(report.error);
  lastReport=report;renderReport();
 }catch(error){lastError=error.message;$('scan-status').textContent=I18n.t('扫描失败');$('scan-note').textContent=error.message;}
 finally{$('scan-button').disabled=false;}
});
fetch('/api/projects/location').then(r=>r.json()).then(data=>{if(!$('project-root').value)$('project-root').value=data.directory;}).catch(()=>{});

function renderReport(){
 const report=lastReport;if(!report)return;$('scan-result').replaceChildren();
  $('scan-status').textContent=report.valid?I18n.msg`${report.project} · 静态校验通过`:I18n.t('规范校验未通过');
  $('scan-note').textContent=I18n.t('静态通过不代表训练已验证。请在项目依赖环境中运行文档中的训练验证命令。通用项目尚不使用 MicroDuck 的网页训练控制。');
  if(!report.valid){const pre=document.createElement('pre');pre.textContent=report.errors.join('\n');$('scan-result').append(pre);return;}
  for(const task of report.tasks){
   const heading=document.createElement('h3');heading.textContent=task.id;
   const facts=document.createElement('p');facts.textContent=I18n.msg`${task.algorithm} · ${task.framework} · ${task.simulator} · ${task.rewards.length} 项奖励`;
   const list=document.createElement('ul');
   for(const reward of task.rewards){const item=document.createElement('li');item.textContent=I18n.msg`${reward.name}：权重 ${reward.weight} · ${reward.enabled?I18n.t('启用'):I18n.t('屏蔽')} · ${task.sources.rewards[reward.name].file}:${task.sources.rewards[reward.name].line}`;list.append(item);}
   $('scan-result').append(heading,facts,list);
  }
}
window.addEventListener('languagechange',()=>{if(lastReport)renderReport();else if(lastError){$('scan-status').textContent=I18n.t('扫描失败');$('scan-note').textContent=I18n.t(lastError);}else if($('scan-button').disabled)$('scan-status').textContent=I18n.t('正在扫描');});
