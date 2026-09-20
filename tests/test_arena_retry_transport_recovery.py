"""One explicitly inspected transport replay; all HTTP is replaced offline."""
import copy
import importlib
import json
from pathlib import Path

import httpx
import pytest

from test_arena_retry_fallback import source, Relay
from scripts import arena_retry_fallback as runner


def recovery_module():
    path = Path(runner.__file__).with_name('arena_retry_transport_recovery.py')
    assert path.is_file(), 'The inspected campaign recovery is not implemented'
    return importlib.import_module('scripts.arena_retry_transport_recovery')


@pytest.fixture
def paused(source, tmp_path):
    campaign = tmp_path / 'campaign'
    runner.prepare_campaign(source, campaign, expected_targets=2)
    with pytest.raises(ValueError, match='Ambiguous'):
        runner.run_campaign(campaign, workers=1, max_requests=1,
                            relay_factory=lambda: Relay([httpx.RemoteProtocolError('private detail')]))
    path = campaign / 'attempts/base/u0-0/02/record.json'
    record = runner.read_json(path)
    trace = [dict(event='attempt_started', audit_id='offline-audit', method='POST',
        request_sha256=record['request_sha256'], transport_attempt=1),
        dict(event='attempt_finished', audit_id='offline-audit', method='POST',
        request_sha256=record['request_sha256'], transport_attempt=1, outcome='exception',
        raised_error_type='RemoteProtocolError', application_headers_started=True,
        connection_failure_phase=None, retry_eligible=False, will_retry=False,
        events=[dict(event='http11.send_request_headers.started', method='POST'),
                dict(event='http11.send_request_headers.complete'),
                dict(event='http11.send_request_body.complete'),
                dict(event='http11.receive_response_headers.failed')])]
    audit=campaign / 'job/connect_attempts.jsonl';audit.parent.mkdir(exist_ok=True)
    audit.write_text(''.join(json.dumps(row)+'\n' for row in trace))
    target=dict(tag='base',uid='u0',order=0,total_attempt=2,
        record_sha256=runner.file_hash(path),local_request_id=record['local_request_id'])
    return campaign,target


def prepared(paused):
    campaign,target=paused
    recovery=recovery_module()
    manifest=recovery.prepare_recovery(campaign,target=target)
    return recovery,campaign,manifest


def test_replay_preserves_old_bytes_has_new_id_and_reuses_logical_attempt_two(paused):
    recovery,campaign,manifest=prepared(paused)
    before={p:p.read_bytes() for p in (campaign/'attempts').rglob('*') if p.is_file()}
    relay=Relay(['[[A>B]]'])
    report=recovery.retry_once(campaign,relay_factory=lambda:relay)
    assert report['resume_eligible'] and report['paid_calls_this_invocation']==1
    assert len(relay.calls)==1 and relay.usage_calls==2
    _,rows=runner.load_resolution(campaign,require_complete=False)
    row=rows[0];proof=row['transport_recovery']
    assert row['resolution']=='gpt4o' and len(row['attempts'])==2
    assert row['selected_record']['total_attempt']==2
    assert row['selected_record']['local_request_id']!=paused[1]['local_request_id']
    assert proof['unknown_usage_requests']==1 and proof['physical_replay_attempts']==1
    assert row['attempts'][1]['record_path']==proof['replacement']['record_path']
    assert all(p.read_bytes()==value for p,value in before.items())
    second=Relay();again=recovery.retry_once(campaign,relay_factory=lambda:second)
    assert again['paid_calls_this_invocation']==0 and second.calls==[]


def test_invalid_replay_then_three_invalid_gpt4o_then_one_fallback(paused):
    recovery,campaign,_=prepared(paused)
    replay=Relay(['invalid'])
    recovery.retry_once(campaign,relay_factory=lambda:replay)
    _,rows=runner.load_resolution(campaign,require_complete=False)
    replacement=rows[0]['attempts'][-1]
    relay=Relay(['invalid']*3+['[[B>A]]','[[A=B]]'])
    result=runner.run_campaign(campaign,workers=1,relay_factory=lambda:relay)
    assert result['complete'] and len(relay.calls)==5
    assert [r['model'] for r in relay.calls[:4]]==['gpt-4o']*3+['gpt-4.1']
    _,rows=runner.load_resolution(campaign)
    assert rows[0]['resolution']=='gpt41' and len(rows[0]['attempts'])==6
    third=runner.read_json(campaign/'attempts/base/u0-0/03/record.json')
    assert third['predecessor_record_sha256']==replacement['record_sha256']
    assert third['supersedes_local_request_id']==replacement['local_request_id']
    assert result['extra_physical_requests']==1 and result['unknown_usage_requests']==1


def test_second_transport_failure_remains_blocked_without_another_replay(paused):
    recovery,campaign,_=prepared(paused)
    relay=Relay([httpx.RemoteProtocolError('private detail')])
    with pytest.raises(ValueError,match='blocked|Ambiguous'):
        recovery.retry_once(campaign,relay_factory=lambda:relay)
    assert len(relay.calls)==1
    with pytest.raises((ValueError,RuntimeError)):
        recovery.retry_once(campaign,relay_factory=lambda:pytest.fail('no second replay'))
    with pytest.raises((ValueError,RuntimeError)):
        runner.run_campaign(campaign,relay_factory=lambda:pytest.fail('blocked chain'))


@pytest.mark.parametrize('damage',['inflight','request','original','extra','authorization','dispatch'])
def test_recovery_damage_blocks_before_network(paused,damage):
    recovery,campaign,manifest=prepared(paused)
    folder=campaign/'transport_recovery'/manifest['recovery_id']
    if damage=='inflight':
        (folder/'inflight.json').write_text('{}')
    elif damage=='extra':
        (folder/'unexpected.json').write_text('{}')
    elif damage=='authorization':
        value=runner.read_json(folder/'manifest.json');value['max_physical_replays']=2
        (folder/'manifest.json').write_text(json.dumps(value))
        (folder/'manifest.sha256').write_text(runner.file_hash(folder/'manifest.json')+'\n')
    elif damage=='original':
        path=campaign/'attempts/base/u0-0/02/record.json';value=runner.read_json(path)
        value['local_request_id']='altered';path.write_text(json.dumps(value))
    elif damage=='dispatch':
        next((campaign/'dispatches').glob('*.json')).write_text('{}')
    else:
        recovery.retry_once(campaign,relay_factory=lambda:Relay())
        path=folder/'record.json';value=runner.read_json(path)
        value['request']['temperature']=1;path.write_text(json.dumps(value))
        (folder/'record.sha256').write_text(runner.file_hash(path)+'\n')
    with pytest.raises((ValueError,RuntimeError)):
        recovery.retry_once(campaign,relay_factory=lambda:pytest.fail('damage must block'))


def test_final_billing_failure_reentry_only_checks_free_billing(paused,monkeypatch):
    recovery,campaign,_=prepared(paused)
    monkeypatch.setattr(runner.time,'sleep',lambda _:None)
    relay=Relay(usages=[150.024436]+[RuntimeError('secret')]*3)
    with pytest.raises(RuntimeError,match='billing'):
        recovery.retry_once(campaign,relay_factory=lambda:relay)
    assert len(relay.calls)==1
    resumed=Relay()
    report=recovery.retry_once(campaign,relay_factory=lambda:resumed)
    assert report['resume_eligible'] and report['paid_calls_this_invocation']==0
    assert resumed.calls==[] and resumed.usage_calls==1


def test_without_authorization_existing_ambiguity_still_blocks(paused):
    with pytest.raises(ValueError,match='Ambiguous'):
        runner.run_campaign(paused[0],relay_factory=lambda:pytest.fail('no implicit recovery'))


def test_wrong_inspected_target_cannot_be_prepared(paused):
    campaign,target=paused;target={**target,'record_sha256':'0'*64}
    with pytest.raises(ValueError):
        recovery_module().prepare_recovery(campaign,target=target)
    assert not (campaign/'transport_recovery').exists()


def test_production_590_scope_rejects_another_inspected_target(paused,monkeypatch):
    campaign,target=paused
    context=runner._load_context(campaign)
    context[1]['target_count']=590
    monkeypatch.setattr(runner,'_load_context',lambda _:context)
    with pytest.raises(ValueError,match='authorized|production|scope'):
        recovery_module().prepare_recovery(campaign,target=target)
    assert not (campaign/'transport_recovery').exists()


def test_unrelated_second_ambiguity_is_not_covered_by_single_recovery(paused):
    campaign,target=paused
    context=runner._load_context(campaign)
    rows,_=runner._resolutions(context,inspected_ambiguity=target)
    pending=next(row for row in rows if row['resolution']=='pending')
    folder,intent=runner._dispatch_intent(context,pending)
    runner._paid_call(folder,intent,Relay([httpx.RemoteProtocolError('second')]),context[3])
    with pytest.raises(ValueError,match='Ambiguous'):
        recovery_module().prepare_recovery(campaign,target=target)
    assert not (campaign/'transport_recovery').exists()


def test_old_completed_campaign_count_schema_remains_readable(source,tmp_path):
    campaign=tmp_path/'old-complete'
    runner.prepare_campaign(source,campaign,expected_targets=2)
    runner.run_campaign(campaign,workers=2,relay_factory=lambda:Relay())
    path=campaign/'complete.json';document=runner.read_json(path)
    document['counts'].pop('blocked',None)
    path.write_text(json.dumps(document))
    _,rows=runner.load_resolution(campaign)
    assert all(row['resolution']=='gpt4o' for row in rows)
