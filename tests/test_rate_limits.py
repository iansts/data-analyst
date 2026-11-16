from unittest.mock import Mock
import pytest
from agent import api_request, ProviderError, rate_limit_details


def response(status, message='', code='', retry=None):
    r = Mock()
    r.status_code = status
    r.headers = {} if retry is None else {'retry-after': str(retry)}
    r.json.return_value = {'error': {'message': message, 'code': code}} if status != 200 else {'ok': True}
    return r


def test_short_limit_retries_identical_request(monkeypatch):
    request = Mock(side_effect=[response(429, 'tokens per minute', retry=2), response(200)])
    sleep = Mock()
    on_wait = Mock()
    monkeypatch.setattr('agent.requests.request', request)
    monkeypatch.setattr('agent.time.sleep', sleep)
    assert api_request('Groq','test','/chat/completions',{'x':1},on_wait) == {'ok': True}
    sleep.assert_called_once_with(2)
    on_wait.assert_called_once_with(2)
    assert request.call_args_list[0] == request.call_args_list[1]


@pytest.mark.parametrize('message,code',[
    ('No credits remaining','insufficient_quota'),
    ('tokens per day (TPD)','rate_limit_exceeded'),
    ('tokens per minute: Limit 8000, Used 0, Requested 9000','rate_limit_exceeded')
])
def test_permanent_limits_do_not_retry(monkeypatch,message,code):
    request = Mock(return_value=response(429,message,code,retry=2))
    monkeypatch.setattr('agent.requests.request',request)
    with pytest.raises(ProviderError):
        api_request('Groq','test','/chat/completions',{})
    assert request.call_count == 1


def test_details_redact_secret_and_parse_wait():
    detail,wait,transient = rate_limit_details(response(429,
        'org_private gsk_secret rate limit. Please try again in 2.5s.'),'gsk_secret')
    assert 'org_private' not in detail and 'gsk_secret' not in detail
    assert wait == 4 and transient


def test_long_wait_is_reported_without_blocking(monkeypatch):
    request = Mock(return_value=response(429,'rate limit',retry=600))
    monkeypatch.setattr('agent.requests.request',request)
    with pytest.raises(ProviderError,match='600'):
        api_request('Groq','test','/chat/completions',{})
    assert request.call_count == 1


def test_retries_are_bounded(monkeypatch):
    request = Mock(return_value=response(429,'rate limit',retry=1))
    monkeypatch.setattr('agent.requests.request',request)
    monkeypatch.setattr('agent.time.sleep',Mock())
    with pytest.raises(ProviderError):
        api_request('Groq','test','/chat/completions',{})
    assert request.call_count == 5


@pytest.mark.parametrize('used,requested,duration,expected',[
    (7184,4002,'23.895s',25),
    (5888,3096,'7.38s',9),
    (5081,2990,'532.5ms',2),
])
def test_actual_groq_tpm_errors_with_billing_link(monkeypatch,used,requested,duration,expected):
    message = (f'Rate limit reached for model `openai/gpt-oss-120b` in organization `org_example` '
               f'service tier `on_demand` on tokens per minute (TPM): Limit 8000, Used {used}, '
               f'Requested {requested}. Please try again in {duration}. Need more tokens? '
               'Upgrade to Dev Tier today at https://console.groq.com/settings/billing')
    limited = response(429,message,'rate_limit_exceeded')
    detail,wait,transient = rate_limit_details(limited,'test-secret')
    assert transient and wait == expected
    assert 'вичерпану квоту' not in detail
    request = Mock(side_effect=[limited,response(200)])
    monkeypatch.setattr('agent.requests.request',request)
    sleep = Mock()
    monkeypatch.setattr('agent.time.sleep',sleep)
    assert api_request('Groq','test-secret','/chat/completions',{}) == {'ok':True}
    sleep.assert_called_once_with(expected)


def test_missing_retry_after_uses_bounded_backoff(monkeypatch):
    request=Mock(side_effect=[response(429,'rate_limit_exceeded'),response(200)])
    monkeypatch.setattr('agent.requests.request',request)
    sleep=Mock()
    monkeypatch.setattr('agent.time.sleep',sleep)
    api_request('Groq','test','/chat/completions',{})
    sleep.assert_called_once_with(10)
