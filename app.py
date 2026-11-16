import hashlib
import json
import re
import traceback
from pathlib import Path

import pandas as pd
import streamlit as st

from agent import APPROVE_PLAN_BUTTON, PLAN_TAB, REJECT_PLAN_BUTTON, Agent, ProviderError
from engine import Clinic, export_csv, export_xlsx, load_table_auto, profile

provider = 'Groq'
model = 'openai/gpt-oss-120b'

st.set_page_config(page_title='Data Analyst · агент якості даних', page_icon='🔬', layout='wide')
st.markdown('''<style>
.stApp {background: #0c1322;}
[data-testid="stSidebar"] {background: #141f32;}
h1,h2,h3 {letter-spacing: -0.025em;}
[data-testid="stMetric"] {background:#17243a;border:1px solid #273a53;border-radius:14px;padding:16px;}
.block-container {padding-top:2.5rem;max-width:1400px;}
[data-testid="stFileUploader"] {border:1px solid #273a53;border-radius:16px;padding:6px;}
[data-testid="stFileUploader"] small {display:none;}
[data-testid="stSidebar"] .stButton button {width:100%;border-radius:10px;}
.stButton button, .stDownloadButton button {border-radius:10px;}
[data-testid="stToolbar"], [data-testid="stAppDeployButton"], #MainMenu {display:none !important;}
</style>''', unsafe_allow_html=True)
st.title('Data Analyst')
st.write('Знаходьте помилки в таблицях і виправляйте їх разом з AI.')


def clear_chat():
    st.session_state.agent = Agent()
    st.session_state.chat = []
    st.session_state.diagnosis_started = False
    if 'clinic' in st.session_state:
        st.session_state.clinic.pending = None


with st.sidebar:
    st.header('Почнімо')
    key = st.text_input('API-ключ', type='password', key='api_key', placeholder='Вставте ключ Groq',
                        help='Ключ зберігається лише протягом поточної сесії.')
    consent = st.checkbox('Дозволяю аналіз даних', help='Зразки таблиці та повідомлення надсилаються Groq для аналізу.')
    st.divider()
    if st.button('Новий діалог', disabled='clinic' not in st.session_state):
        clear_chat()
        st.rerun()
    if st.button('Завершити сесію'):
        for k in list(st.session_state):
            del st.session_state[k]
        st.rerun()

upload = st.file_uploader('Завантажте таблицю', type=['csv','xlsx'],
                          disabled=not key.strip() or not consent)
st.caption('CSV · XLSX')

if not key.strip():
    st.info('Введіть API-ключ зліва, щоб почати.')
    st.stop()
if not consent:
    st.stop()

if upload is not None:
    data = upload.getvalue()
    identity = hashlib.sha256(upload.name.encode() + b'\0' + data).hexdigest()
    if identity != st.session_state.get('loaded_file'):
        try:
            df = load_table_auto(data, upload.name)
            st.session_state.clinic = Clinic(df)
            st.session_state.filename = upload.name
            st.session_state.loaded_file = identity
            clear_chat()
        except Exception as exc:
            st.error(f'Не вдалося прочитати таблицю: {str(exc)[:250]}')
            st.stop()

if 'clinic' not in st.session_state:
    st.stop()

clinic = st.session_state.clinic
agent = st.session_state.agent
if type(agent) is not Agent:
    # Hot reload: keep the conversation while adopting the updated agent code.
    updated_agent = Agent()
    updated_agent.messages = agent.messages
    updated_agent.trace = agent.trace
    updated_agent.total_tokens = agent.total_tokens
    st.session_state.agent = agent = updated_agent
stats = profile(clinic.df)
st.caption(st.session_state.filename)
cols = st.columns(4)
for c, label, val in zip(cols, ['Рядків','Колонок','Пропущених клітинок','Повних дублів'],
                         [stats['rows'],stats['columns'],stats['missing_cells'],stats['duplicate_rows']]):
    c.metric(label,val)


def talk(text):
    if key.strip() and key.strip() in text:
        st.error('Не надсилайте ключ у чат. Він має бути лише в прихованому полі.')
        return
    st.session_state.chat.append({'role': 'user', 'content': text})
    try:
        with st.spinner('Думаю…', show_time=False):
            # Automatic retries remain inside the agent. Keep one calm indicator
            # throughout network calls, tool execution and rate-limit backoff.
            # Detailed successful tool calls remain available in the report tab.
            result = agent.run(clinic, text, provider, key, model)
        st.session_state.chat.append({'role':'assistant','content':result})
    except ProviderError as exc:
        st.session_state.chat.append({'role':'assistant','content':f'⚠️ {exc} Запит не завершено; можна повторити.'})
    except Exception as exc:
        detail = f'{type(exc).__name__}: {exc}'.replace(key.strip(), '[ключ приховано]')
        detail = re.sub(r'\b(?:gsk_|xai-|sk-|org_)[A-Za-z0-9_-]+', '[приховано]', detail)
        frames = traceback.extract_tb(exc.__traceback__)
        location = ' → '.join(f'{Path(f.filename).name}:{f.lineno} ({f.name})' for f in frames)
        # No locals, request headers, credentials or raw API responses in diagnostics.
        print(f'Data Analyst failure: {detail[:600]} | {location}', flush=True)
        st.session_state.chat.append({'role':'assistant','content':f'⚠️ Помилка виконання: {detail[:600]}\n\nПоточна версія таблиці: {clinic.revision}.\n\nМісце: {location}'})
    st.rerun()


chat_tab, data_tab, plan_tab, log_tab = st.tabs(['Розмова з агентом','Таблиця й аналіз',PLAN_TAB,'Звіт та експорт'])
with chat_tab:
    st.subheader('Дослідження')
    if st.button('Почати діагностику'):
        st.session_state.diagnosis_started = True
        talk('Досліди якість таблиці інструментами, перевір можливі проблеми та запитай мене про неоднозначності перед виправленнями.')
    for message in st.session_state.chat:
        with st.chat_message(message['role']):
            st.markdown(message['content'])
    if clinic.pending:
        st.info(f'План готовий. Відкрийте «{PLAN_TAB}», перегляньте зміни й натисніть '
                f'«{APPROVE_PLAN_BUTTON}» або «{REJECT_PLAN_BUTTON}».')
    if st.session_state.get('diagnosis_started', False):
        prompt = st.chat_input('Поставте задачу або дайте відповідь агенту', max_chars=5000)
        if prompt:
            talk(prompt)

with data_tab:
    st.subheader('Поточна таблиця')
    st.dataframe(clinic.df, use_container_width=True)
    with st.expander('Оригінал'):
        st.dataframe(clinic.original, use_container_width=True)
    st.subheader('Пропуски за колонками')
    st.bar_chart(clinic.df.isna().sum())
    numeric = list(clinic.df.select_dtypes(include='number').columns)
    if numeric:
        col = st.selectbox('Числова колонка', numeric)
        st.line_chart(clinic.df[[col]].reset_index(drop=True))
        st.caption('Значення за порядком рядків. Графік не означає часову залежність.')

with plan_tab:
    st.subheader('Перевірте зміни')
    if clinic.pending:
        p = clinic.pending['preview']
        st.write(p['explanation'])
        st.write(f"Зміниться рядків: **{p['changed_rows']}** · буде видалено: **{p['removed_rows']}**")
        with st.expander('Деталі змін'):
            st.json(p['operations'])
        left,right = st.columns(2)
        left.write('До · до 30 змінених рядків')
        left.dataframe(pd.DataFrame(p['before_sample']), use_container_width=True)
        right.write('Після')
        right.dataframe(pd.DataFrame(p['after_sample']), use_container_width=True)
        if p['removed_rows']:
            st.write('Рядки, які буде видалено · перші 30')
            st.dataframe(pd.DataFrame(p['removed_sample']))
        with st.expander('Повна таблиця після змін'):
            st.dataframe(clinic.pending['candidate'], use_container_width=True)
        if st.button(APPROVE_PLAN_BUTTON, type='primary'):
            clinic.approve(p['id'])
            talk(f"Я затвердив(-ла) план {p['id']} кнопкою. Застосовано версію {clinic.revision}. Повторно перевір результат інструментами; вкажи невирішені проблеми.")
        if st.button(REJECT_PLAN_BUTTON):
            clinic.reject()
            agent.messages.append({'role':'user','content':f"План {p['id']} відхилено кнопкою. Не застосований."})
            st.rerun()
    else:
        st.info('Коли агент підготує план, тут з’являться операції й попередній перегляд.')
    if st.button('Відкотити останнє застосування', disabled=not clinic.snapshots):
        clinic.undo()
        agent.messages.append({'role':'user','content':f'Виконано відкат. Поточна версія {clinic.revision}. Перед висновками перевір дані знову.'})
        st.rerun()
    st.caption('Доступні останні 5 станів. Закриття сесії очищає дані; експортуйте результат перед виходом.')

with log_tab:
    st.subheader('Збережіть результат')
    st.download_button('Таблиця CSV', export_csv(clinic.df), 'cleaned.csv', 'text/csv')
    st.download_button('Таблиця Excel', export_xlsx(clinic.df), 'cleaned.xlsx',
                       'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    report = clinic.report() | {'conversation':st.session_state.chat, 'tool_calls':agent.trace}
    st.download_button('Звіт JSON із журналом', json.dumps(report,ensure_ascii=False,indent=2,default=str).encode(), 'report.json','application/json')
    st.caption(f'Дій агента: {len(agent.trace)}')
    for i, item in enumerate(reversed(agent.trace)):
        with st.expander(f"{len(agent.trace)-i}. {item['tool']}"):
            st.json(item)
    with st.expander('Історія застосувань і відкатів'):
        st.json(clinic.audit)
