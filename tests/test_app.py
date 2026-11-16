from pathlib import Path
from streamlit.testing.v1 import AppTest
from engine import Clinic
from agent import Agent
import pandas as pd

APP = str(Path(__file__).parents[1] / 'app.py')

def test_no_key_gates_app():
    app = AppTest.from_file(APP).run()
    assert not app.exception
    assert not app.chat_input
    assert any('API-ключ' in x.value for x in app.info)

def test_loaded_workspace_renders_without_network():
    app = AppTest.from_file(APP)
    app.session_state['api_key'] = 'test-only-never-sent'
    app.session_state['clinic'] = Clinic(pd.DataFrame({'city':[' A ',' A '], 'value':[1,1]}))
    app.session_state['agent'] = Agent()
    app.session_state['chat'] = []
    app.session_state['filename'] = 'test.csv'
    app.run()
    app.checkbox[0].check().run()
    assert not app.exception
    assert len(app.metric) == 4
    app.session_state['clinic'].propose_plan([{'kind':'trim','column':'city'}],'Trim')
    app.run()
    assert not app.exception
    assert any('Підтверджую' in b.label for b in app.button)
