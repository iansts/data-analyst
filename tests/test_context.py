import json
import pandas as pd
from agent import Agent, SYSTEM, RequestTooLarge, compact_history, shrink_tool_results
from engine import Clinic


def test_compaction_preserves_all_clarifications_and_approved_state():
    c=Clinic(pd.DataFrame({'city':[' A ']}))
    p=c.propose_plan([{'kind':'trim','column':'city'}],'Confirmed')
    c.approve(p['id'])
    messages=[{'role':'system','content':SYSTEM}]
    for i in range(15):
        messages.extend([
            {'role':'user','content':f'Уточнення {i}: не видаляй повернення.'},
            {'role':'assistant','content':None,'tool_calls':[{'id':str(i)}]},
            {'role':'tool','tool_call_id':str(i),'content':'x'*5000},
            {'role':'assistant','content':f'Відповідь {i}'}])
    compact=compact_history(messages,c)
    assert [m for m in compact if m['role']=='user']==[m for m in messages if m['role']=='user']
    assert not any(m['role']=='tool' or m.get('tool_calls') for m in compact)
    assert compact[-1]['content']=='Відповідь 14'
    assert 'approved_plan' in compact[0]['content'] and p['id'] in compact[0]['content']
    assert len(json.dumps(compact)) < len(json.dumps(messages))/5


def test_pending_and_rejected_plan_survive_compaction():
    c=Clinic(pd.DataFrame({'city':[' A ']}))
    p=c.propose_plan([{'kind':'trim','column':'city'}],'Trim')
    assert p['id'] in compact_history([],c)[0]['content']
    c.reject()
    context=compact_history([],c)[0]['content']
    assert 'rejected_plan' in context and 'trim' in context


def test_retiring_results_keeps_call_ids_and_latest_results():
    messages=[]
    for i in range(5):
        messages.extend([{'role':'assistant','tool_calls':[{'id':str(i)}]},
                         {'role':'tool','tool_call_id':str(i),'content':str(i)*3000}])
    small=shrink_tool_results(messages)
    assert [m['tool_call_id'] for m in small if m['role']=='tool']==list(map(str,range(5)))
    assert small[-1]==messages[-1] and small[-3]==messages[-3]
    assert len(json.dumps(small)) < len(json.dumps(messages))


def test_oversize_retry_does_not_mutate_approved_table(monkeypatch):
    c=Clinic(pd.DataFrame({'city':[' A ']}))
    p=c.propose_plan([{'kind':'trim','column':'city'}],'Trim');c.approve(p['id'])
    sent=[]
    def fake(provider,key,path,payload,**kwargs):
        sent.append(payload.copy())
        if len(sent)==1:
            raise RequestTooLarge('oversized')
        return {'choices':[{'message':{'content':'Перевірено'}}]}
    monkeypatch.setattr('agent.api_request',fake)
    a=Agent()
    assert a.run(c,'Перевір після затвердження','Groq','test','model')=='Перевірено'
    assert c.revision==1 and c.df.iloc[0,0]=='A'
    assert len(c.audit)==1 and len(sent)==2
    assert sent[1]['max_tokens']==1000
