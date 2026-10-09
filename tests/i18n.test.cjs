const test=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const path=require('node:path');
const staticRoot=path.join(__dirname,'../rl_workbench/static');
function load(saved='zh-CN',blocked=false){
 const events={},text={nodeValue:'训练监控',isConnected:true,parentElement:{closest:()=>false}};
 const field={value:'未保存的名称',defaultValue:'默认名称'};
 const select={value:'',add(){},setAttribute(){},addEventListener(type,fn){this.change=fn;}};
 const placeholder={isConnected:true,attrs:{placeholder:'例如 256, 128'},getAttribute(k){return this.attrs[k]??null;},setAttribute(k,v){this.attrs[k]=v;}};
 let visited=false;
 const document={documentElement:{lang:''},
  createTreeWalker(){return {nextNode(){if(visited)return false;visited=true;this.currentNode=text;return true;}};},
  querySelectorAll(query){return query.startsWith('[title]')?[placeholder]:[field];},
  querySelector(){return {append(){}};},
  createElement(tag){return tag==='select'?select:{dataset:{},append(){}};},
  createTextNode(t){return {nodeValue:t};},getElementById(){return select;}};
 const store=new Map([['rl-workbench-language',saved]]);
 const context={document,NodeFilter:{SHOW_TEXT:4},Option:function(text,value){this.text=text;this.value=value;},CustomEvent:function(type){this.type=type;},
  localStorage:{getItem(k){if(blocked)throw Error('blocked');return store.get(k);},setItem(k,v){if(blocked)throw Error('blocked');store.set(k,v);}},
  addEventListener(name,fn){events[name]=fn;},dispatchEvent(event){events[event.type]?.(event);}};
 context.window=context;vm.createContext(context);
 for(const name of ['translations.js','i18n.js'])vm.runInContext(fs.readFileSync(path.join(staticRoot,name),'utf8'),context);
 return {...context,text,field,placeholder,store,events};
}
test('switching updates static text, accessible hints and persistence without resetting a draft',()=>{
 const ui=load();ui.I18n.set('en');
 assert.equal(ui.text.nodeValue,'Training monitor');
 assert.equal(ui.document.documentElement.lang,'en-US');
 assert.equal(ui.placeholder.attrs.placeholder,'Example:  256, 128');
 assert.equal(ui.field.value,'未保存的名称');
 assert.equal(ui.store.get('rl-workbench-language'),'en');
 ui.I18n.set('zh-CN');assert.equal(ui.text.nodeValue,'训练监控');
});
test('dynamic text translates only literal parts; names and identifiers are preserved',()=>{
 const ui=load('en');const name='训练监控 <user-owned> ${value}';
 const result=ui.I18n.msg(['删除训练记录“','”（', '）？'],name,'abc123');
 assert.equal(result,`Delete training record “${name}” (abc123)?`);
 assert.equal(ui.I18n.t('训练记录及其文件已删除'),'Training record and files deleted');
 const labels=ui.I18n.labels({running:'训练中'});assert.equal(labels.running,'Training');
 ui.I18n.set('zh-CN');assert.equal(labels.running,'训练中');
});
test('stored language loads on next page and storage events synchronize without writing',()=>{
 const ui=load('en');assert.equal(ui.text.nodeValue,'Training monitor');
 ui.events.storage({key:'rl-workbench-language',newValue:'zh-CN'});
 assert.equal(ui.text.nodeValue,'训练监控');
 assert.equal(ui.store.get('rl-workbench-language'),'en');
});
test('unavailable localStorage does not break the UI or language switching',()=>{
 const ui=load('en',true);assert.equal(ui.I18n.language,'zh-CN');
 ui.I18n.set('en');assert.equal(ui.I18n.t('准备就绪'),'Ready');
});
test('all shipped UI Chinese fragments have an English dictionary entry',()=>{
 const ui=load();
 for(const name of ['microduck.html','index.html','projects.html','microduck.js','app.js','projects.js']){
  const source=fs.readFileSync(path.join(staticRoot,name),'utf8');
  for(const phrase of source.match(/[\u3400-\u9fff，。；、：“”「」（）…！？]+/g)||[]){
   assert.ok(Object.hasOwn(ui.RL_TRANSLATIONS,phrase),`${name}: ${phrase}`);
  }
 }
});
