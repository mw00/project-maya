"""Exercise the real Qwen setup preservation helper and its actual flag list."""
import ast
from pathlib import Path
s=(Path(__file__).resolve().parents[1]/'maya.py').read_text();tree=ast.parse(s)
helper=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='keep_args')
writer=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='write_config_qwen')
call=next(n for n in ast.walk(writer) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='keep_args')
flags=ast.literal_eval(call.args[2]);ns={};exec(compile(ast.Module(body=[helper],type_ignores=[]),'actual-helper','exec'),ns)
values={'--rope-scaling':'yarn','--rope-scale':'2','--rope-freq-base':'10000000','--rope-freq-scale':'0.5','--yarn-orig-ctx':'262144','--yarn-ext-factor':'1','--yarn-attn-factor':'1','--yarn-beta-fast':'32','--yarn-beta-slow':'1'}
old=[x for kv in values.items() for x in kv];new=['--max-context','524288'];ns['keep_args'](old,new,flags)
for flag,val in values.items():assert new[new.index(flag)+1]==val
new=['--rope-scaling','yarn'];ns['keep_args'](['--rope-scaling','none'],new,flags);assert new==['--rope-scaling','none']
print('PASS: actual Qwen setup preserves all scaling args, including explicit none')
