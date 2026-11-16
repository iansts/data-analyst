import io
import json
import pandas as pd
import pytest
from openpyxl import load_workbook
from engine import Clinic, transform, load_table, export_csv, export_xlsx
from agent import Agent, ProviderError
from engine import load_table_auto


@pytest.mark.parametrize('sep', [',',';','\t','|'])
def test_automatic_csv_format(sep):
    data=f'city{sep}amount\nКиїв{sep}"120,50"\nЛьвів{sep}"30,00"\n'
    for encoding in ('utf-8-sig','cp1251'):
        df=load_table_auto(data.encode(encoding),'data.csv')
        assert list(df.columns)==['city','amount']
        assert df.iloc[0].tolist()==['Київ','120,50']


def test_preview_approval_undo():
    original = pd.DataFrame({'city':[' Київ ',' Київ '], 'price':['2,50','2,50']})
    c = Clinic(original)
    p = c.propose_plan([{'kind':'trim','column':'city'},
                        {'kind':'to_number','column':'price','decimal':','},
                        {'kind':'drop_duplicates'}], 'Normalize')
    pd.testing.assert_frame_equal(c.df, original)
    assert p['removed_rows'] == 1
    assert p['changed_rows'] == 1
    with pytest.raises(ValueError):
        c.approve('wrong-plan')
    c.approve(p['id'])
    assert c.df.iloc[0]['price'] == 2.5 and len(c.df) == 1
    c.undo()
    pd.testing.assert_frame_equal(c.df, original)
    assert c.pending is None


def test_failed_plan_is_atomic():
    c = Clinic(pd.DataFrame({'v':[' a ','bad']}))
    with pytest.raises(ValueError):
        c.propose_plan([{'kind':'trim','column':'v'}, {'kind':'to_number','column':'v'}], 'x')
    assert c.df.iloc[0,0] == ' a '
    assert c.pending is None


def test_dates_require_explicit_valid_format():
    df = pd.DataFrame({'d':['31/12/2026','12/31/2026']})
    with pytest.raises(ValueError):
        transform(df,[{'kind':'to_date','column':'d','format':'%d/%m/%Y'}])
    with pytest.raises(ValueError):
        transform(df,[{'kind':'to_date','column':'d','format':'mixed'}])


def test_formula_safe_export():
    df = pd.DataFrame({'x':['=1+1','@SUM(A1)','normal'], 'n':[-2,3,4]})
    assert "'=1+1" in export_csv(df).decode('utf-8-sig')
    wb = load_workbook(io.BytesIO(export_xlsx(df)))
    assert wb.active['A2'].data_type == 's'
    assert wb.active['B2'].value == -2


def test_read_csv_and_xlsx():
    df = load_table(b'name;value\nA;2\n', 'data.csv', sep=';')
    assert df.iloc[0]['value'] == 2
    reread = load_table(export_xlsx(df), 'data.xlsx')
    pd.testing.assert_frame_equal(df, reread)


def test_literal_search_and_numeric_aggregation():
    c = Clinic(pd.DataFrame({'g':['[x]','abc'], 'n':[2,3]}))
    assert c.inspect_rows('g','[')['matched'] == 1
    assert c.aggregate('g','n')['groups'] == 2


def test_tool_loop_and_rejection_of_unavailable_tool(monkeypatch):
    responses = iter([
        {'choices':[{'message': {'content':None,'tool_calls':[
            {'id':'a','type':'function','function':{'name':'profile_table','arguments':'{}'}},
            {'id':'b','type':'function','function':{'name':'approve','arguments':'{}'}}]}}]},
        {'choices':[{'message':{'content':'Готово'}}]}
    ])
    monkeypatch.setattr('agent.api_request', lambda *a,**kw: next(responses))
    c = Clinic(pd.DataFrame({'x':[1,1]}))
    a = Agent()
    assert a.run(c,'check','Groq','test','model') == 'Готово'
    assert a.trace[0]['result']['duplicate_rows'] == 1
    assert 'error' in a.trace[1]['result']
    assert c.revision == 0
    assert [m['role'] for m in a.messages] == ['system','user','assistant','tool','tool','assistant']


def test_provider_failure_rolls_back_plan_and_history(monkeypatch):
    calls = 0
    def fake(*args,**kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ProviderError('offline')
        return {'choices':[{'message':{'content':None,'tool_calls':[
            {'id':'a','type':'function','function':{'name':'propose_plan', 'arguments':json.dumps(
                {'operations':[{'kind':'trim','column':'x'}], 'explanation':'x'})}}
        ]}}]}
    monkeypatch.setattr('agent.api_request',fake)
    c = Clinic(pd.DataFrame({'x':[' a ']}))
    a = Agent()
    with pytest.raises(ProviderError):
        a.run(c,'check','Groq','test','model')
    assert c.pending is None and len(a.messages) == 1
    assert c.df.iloc[0,0] == ' a '
