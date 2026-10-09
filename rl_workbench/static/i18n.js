'use strict';
// Translate only application-authored text. Template values (names, paths, IDs)
// are interpolated afterwards and are never passed through the dictionary.
window.I18n = (() => {
 const dictionary=window.RL_TRANSLATIONS;
 const phrase=/[\u3400-\u9fff，。；、：“”「」（）…！？]+/g;
 const key='rl-workbench-language';
 let language='zh-CN';
 try{if(localStorage.getItem(key)==='en')language='en';}catch{}
 const textNodes=[],attributes=[];
 const t=text=>language==='en'?(dictionary[String(text)]??String(text).replace(phrase,part=>dictionary[part]??part)):String(text);
 const api={
  t,
  msg(strings,...values){
   const key=strings.reduce((result,part,i)=>result+part+(i<values.length?`{${i}}`:''),'');
   const translated=window.RL_MESSAGE_TRANSLATIONS?.[key];
   if(language==='en'&&translated)return translated.replace(/\{(\d+)\}/g,(_,index)=>String(values[Number(index)]));
   return strings.reduce((result,part,i)=>result+t(part)+(i<values.length?values[i]:''),'');
  },
  labels(values){return new Proxy(values,{get(target,name){return typeof target[name]==='string'?t(target[name]):target[name];}});},
  get locale(){return language==='en'?'en-US':'zh-CN';},
  get language(){return language;},
  set(next,persist=true){
   if(!['zh-CN','en'].includes(next))return;
   language=next;
   if(persist)try{localStorage.setItem(key,next);}catch{}
   apply();
   window.dispatchEvent(new CustomEvent('languagechange'));
  }
 };
 function apply(){
  document.documentElement.lang=api.locale;
  for(const [node,original] of textNodes)if(node.isConnected)node.nodeValue=t(original);
  for(const [node,attribute,original] of attributes)if(node.isConnected)node.setAttribute(attribute,t(original));
  const selector=document.getElementById('ui-language');if(selector)selector.value=language;
 }
 // This script is deferred and precedes page scripts: capture only static HTML.
 const walker=document.createTreeWalker(document.documentElement,NodeFilter.SHOW_TEXT);
 while(walker.nextNode()){
  const node=walker.currentNode;
  if(!node.parentElement.closest('script,style,pre,code,[data-no-i18n]')&&/[\u3400-\u9fff]/.test(node.nodeValue))textNodes.push([node,node.nodeValue]);
 }
 for(const node of document.querySelectorAll('[title],[aria-label],[placeholder],[alt],meta[name="description"]')){
  for(const attribute of ['title','aria-label','placeholder','alt','content']){
   const original=node.getAttribute(attribute);
   if(original&&/[\u3400-\u9fff]/.test(original))attributes.push([node,attribute,original]);
  }
 }
 for(const input of document.querySelectorAll('input[type="text"],input:not([type])')){
  if(input.value===input.defaultValue)input.value=t(input.value);
 }
 const languageControl=document.createElement('label');languageControl.className='language-control';languageControl.dataset.noI18n='true';
 languageControl.append(document.createTextNode('语言 / Language '));
 const select=document.createElement('select');select.id='ui-language';select.setAttribute('aria-label','语言 / Language');
 select.add(new Option('中文','zh-CN'));select.add(new Option('English','en'));languageControl.append(select);
 (document.querySelector('.sidebar')||document.body).append(languageControl);
 select.addEventListener('change',()=>api.set(select.value));
 window.addEventListener('storage',event=>{if(event.key===key)api.set(event.newValue==='en'?'en':'zh-CN',false);});
 apply();return api;
})();
