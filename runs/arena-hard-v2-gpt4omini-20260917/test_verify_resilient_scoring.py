"""Independent bounded recovery validation; no network or model execution."""
import copy
import importlib.util
import json
from pathlib import Path
import sys

import pytest

SUITE = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def modules():
    return (load('independent_resilient_verifier', SUITE / 'verify_resilient_scoring.py'),
            load('independent_resilient_base', SUITE / 'verify_final_scoring.py'),
            load('independent_resilient_judge', SUITE / 'source/scripts/judge_arena_hard.py'))


def fixture_chain(tmp_path, kinds=('ConnectError', 'structural'), *, legacy=True):
    api, base, judge = modules()
    protocol = judge.load_protocol(SUITE / 'source/third_party/arena_hard', baseline_model=base.BASELINE)
    request = {'model': 'gpt-4.1', 'temperature': 0., 'max_tokens': 16000,
               'messages': [{'role': 'system', 'content': 'official system'},
                            {'role': 'user', 'content': 'new reference and candidate'}]}
    state = tmp_path / 'model_judgment/gpt-4.1/state'
    policy = {'policy': 'arena_bounded_connection_recovery_v1', 'declared_at': '2026-09-17T12:00:00Z',
              'max_total_attempts_per_game': 5, 'transport_max_connect_attempts': 4,
              'allowed_transport_errors': ['ConnectError', 'ConnectTimeout'],
              'reviewed_existing_failures': {}}
    binding = {'policy': policy, 'policy_sha256': 'p' * 64, 'helper_sha256': 'h' * 64, 'suite': tmp_path}
    rows = []
    for i, kind in enumerate((*kinds, 'valid')):
        timestamp = '2026-09-17T11:59:00Z' if legacy and i == 0 else f'2026-09-17T12:0{i}:00Z'
        row = {'tag': 'sft-init', 'uid': 'u1', 'order': 0, 'attempt': i,
               'local_request_id': f'request-{i}', 'request': request,
               'request_sha256': base.digest(request), 'baseline_model': base.BASELINE,
               'protocol_sha256': base.digest(protocol), 'started_at': timestamp, 'finished_at': timestamp}
        if i:
            row.update(supersedes_local_request_id=f'request-{i-1}',
                       resilient_policy_sha256=binding['policy_sha256'],
                       resilient_helper_sha256=binding['helper_sha256'])
        if kind == 'valid':
            row.update(status='valid', answer='[[B>A]]', score='B>A', finish_reason='stop',
                       usage={'prompt_tokens': 3, 'completion_tokens': 4, 'total_tokens': 7})
        elif kind == 'structural':
            row.update(status='invalid', answer='missing verdict', score=None, finish_reason='length',
                       usage={'prompt_tokens': 3, 'completion_tokens': 4, 'total_tokens': 7})
        else:
            row.update(status='ambiguous', error_type=kind)
        rows.append(row)
    archive_dir = state / 'attempts/sft-init/u1-0'; archive_dir.mkdir(parents=True)
    for row in rows[:-1]:
        path = archive_dir / (row['local_request_id'] + '.json')
        path.write_text(json.dumps(row))
    if legacy:
        policy['reviewed_existing_failures']['sft-init:u1:0'] = {
            'path': 'model_judgment/gpt-4.1/state/games/sft-init/u1-0.json',
            'sha256': base.file_hash(archive_dir / 'request-0.json')}
    return api, base, judge, request, protocol, binding, state, rows, archive_dir


def verify_fixture(f):
    api, base, judge, request, protocol, binding, state, rows, _ = f
    return api.verify_resilient_chain(base, rows[-1], state, 'sft-init', 'u1', 0, request,
                                     judge.parse_score, binding, protocol)


def test_mixed_whitelisted_connection_and_structural_chain_passes(tmp_path):
    f = fixture_chain(tmp_path)
    result = verify_fixture(f)
    assert len(result['archives']) == 2 and len(result['request_ids']) == 3
    assert result['connect_failures'] == 1 and result['structural_invalid_attempts'] == 1
    assert result['initial_failure_kind'] == 'connection'


@pytest.mark.parametrize('error', ['ConnectError', 'ConnectTimeout'])
def test_new_connection_failure_is_allowed_only_after_declaration(tmp_path, error):
    f = fixture_chain(tmp_path, (error,), legacy=False)
    assert verify_fixture(f)['connect_failures'] == 1


@pytest.mark.parametrize('damage', ['read_timeout', 'write_timeout', 'protocol_error', 'http_error',
    'inflight', 'response_answer', 'response_id', 'usage_none', 'valid_winner', 'valid_tie',
    'missing_allowlist', 'allowlist_hash', 'allowlist_path', 'wrong_request', 'wrong_request_hash',
    'baseline', 'protocol', 'helper', 'policy', 'missing_helper', 'predates_policy',
    'skip', 'cycle', 'traversal', 'missing_archive', 'bad_archive_id', 'before_prior_finish', 'six_attempts'])
def test_unsafe_or_unbound_chain_rejected(tmp_path, damage):
    f = fixture_chain(tmp_path, ('ConnectError',)*5 if damage == 'six_attempts' else ('ConnectError','structural'))
    api, base, judge, request, protocol, binding, state, rows, directory = f
    old = rows[0]
    if damage in ['read_timeout','write_timeout','protocol_error','http_error']:
        old['error_type'] = {'read_timeout':'ReadTimeout','write_timeout':'WriteTimeout',
                            'protocol_error':'RemoteProtocolError','http_error':'HTTPStatusError'}[damage]
    elif damage == 'inflight': old['status'] = 'inflight'
    elif damage == 'response_answer': old['answer'] = 'partial response'
    elif damage == 'response_id': old['response_id'] = 'provider-id'
    elif damage == 'usage_none': old['usage'] = None
    elif damage in ['valid_winner','valid_tie']:
        old.update(status='valid',answer='[[A=B]]' if damage=='valid_tie' else '[[A>B]]',
                   score='A=B' if damage=='valid_tie' else 'A>B',finish_reason='stop')
    elif damage == 'missing_allowlist': binding['policy']['reviewed_existing_failures'] = {}
    elif damage == 'allowlist_hash': binding['policy']['reviewed_existing_failures']['sft-init:u1:0']['sha256']='0'*64
    elif damage == 'allowlist_path': binding['policy']['reviewed_existing_failures']['sft-init:u1:0']['path']='other.json'
    elif damage == 'wrong_request': old['request'] = {**request,'max_tokens':32000}
    elif damage == 'wrong_request_hash': old['request_sha256']='0'*64
    elif damage == 'baseline': old['baseline_model']='o3-mini-2025-01-31'
    elif damage == 'protocol': old['protocol_sha256']='0'*64
    elif damage == 'helper': rows[-1]['resilient_helper_sha256']='x'*64
    elif damage == 'policy': rows[-1]['resilient_policy_sha256']='x'*64
    elif damage == 'missing_helper': rows[-1].pop('resilient_helper_sha256')
    elif damage == 'predates_policy': rows[-1]['started_at']='2026-09-17T11:00:00Z'
    elif damage == 'skip': rows[-1]['attempt']=4
    elif damage == 'cycle': rows[-1]['supersedes_local_request_id']=rows[-1]['local_request_id']
    elif damage == 'traversal': rows[-1]['supersedes_local_request_id']='../request-1'
    elif damage == 'missing_archive': (directory/'request-0.json').unlink()
    elif damage == 'bad_archive_id': old['local_request_id']='mismatched'
    elif damage == 'before_prior_finish': rows[1]['finished_at']='2026-09-17T12:59:00Z'
    if damage != 'missing_archive':
        (directory/'request-0.json').write_text(json.dumps(old))
        # Keep byte whitelist valid when testing independent semantic rejections.
        if damage not in ['missing_allowlist','allowlist_hash','allowlist_path']:
            binding['policy']['reviewed_existing_failures']['sft-init:u1:0']['sha256']=base.file_hash(directory/'request-0.json')
    (directory/'request-1.json').write_text(json.dumps(rows[1]))
    with pytest.raises(ValueError):verify_fixture(f)


@pytest.fixture(scope='module')
def complete_suite(tmp_path_factory):
    """Real frozen inputs and preserved pilot bytes; all new calls are synthetic."""
    import datetime
    import shutil
    required = ('experiment.json', 'host_execution.json', 'manual_transport_recovery.json',
                'model_judgment/gpt-4.1/state/billing.json',
                'job/transport_incident/host_execution.json')
    missing = [name for name in required if not (SUITE/name).is_file()]
    if missing:
        pytest.skip('Requires local pre-recovery Arena artifacts: ' + ', '.join(missing[:3]))
    experiment = json.loads((SUITE/'experiment.json').read_text())
    artifact_names = [name for name in experiment['files_sha256']
                      if not name.startswith('source/') and Path(name).suffix not in ('.py', '.sh')]
    missing = [name for name in artifact_names if not (SUITE/name).is_file()]
    if missing:
        pytest.skip('Requires local archived answers and provenance: ' + ', '.join(missing[:3]))
    records = [json.loads(path.read_text()) for path in
               (SUITE/'model_judgment/gpt-4.1/state/games').glob('*/*.json')]
    valid_count = sum(row['status'] == 'valid' for row in records)
    if len(records) != 24 or valid_count != 22:
        pytest.skip('Requires the pre-recovery snapshot with 22 valid and 2 unresolved games; '
                    'the live evaluation has advanced')
    suite = tmp_path_factory.mktemp('complete-resilient-qa')
    for name in ('source', 'reference', 'original_generation', '.tiktoken-cache', 'manifests'):
        (suite/name).symlink_to(SUITE/name, target_is_directory=True)
    shutil.copytree(SUITE/'model_answer', suite/'model_answer')
    for source in SUITE.iterdir():
        if source.is_file() and source.suffix in ('.py', '.json', '.jsonl', '.sh'):
            shutil.copyfile(source, suite/source.name)
    shutil.copytree(SUITE/'model_judgment', suite/'model_judgment')
    shutil.copytree(SUITE/'job/transport_incident', suite/'job/transport_incident')
    api = load('complete_resilient_api', suite/'verify_resilient_scoring.py')
    base = load('complete_resilient_base', suite/'verify_final_scoring.py')
    controller = load('complete_resilient_controller', suite/'continue_evaluation.py')
    judge = controller.judge_module(suite)
    run = controller.load_judge_run(suite)
    state = run.directory/'state'
    current = list((state/'games').glob('*/*.json'))
    preserved = {str(p.relative_to(suite)):base.file_hash(p) for p in current if json.loads(p.read_text())['status']=='valid'}
    assert len(preserved)==22
    unresolved = {':'.join(map(str,(r['tag'],r['uid'],r['order']))):
                  {'path':str(p.relative_to(suite)),'sha256':base.file_hash(p)}
                  for p in current for r in [json.loads(p.read_text())] if r['status']!='valid'}
    assert len(unresolved)==2
    declared = datetime.datetime.now(datetime.timezone.utc)
    begin = (declared + datetime.timedelta(seconds=1)).isoformat()
    end = (declared + datetime.timedelta(seconds=2)).isoformat()
    for name in api.REQUIRED_SOURCES:
        p=suite/name
        if not p.exists():p.write_text('# offline pinned placeholder; no execution\n')
    policy = {'policy':api.POLICY,'declared_at':declared.isoformat(),
              'max_total_attempts_per_game':5,'transport_max_connect_attempts':4,
              'allowed_transport_errors':['ConnectError','ConnectTimeout'],
              'original_experiment_sha256':base.file_hash(suite/'experiment.json'),
              'source_sha256':{n:base.file_hash(suite/n) for n in api.REQUIRED_SOURCES},
              'preserved_valid_records_sha256':preserved,'reviewed_existing_failures':unresolved,
              'original_host_execution_sha256':base.file_hash(suite/'host_execution.json'),
              'legacy_manual_approval_sha256':base.file_hash(suite/'manual_transport_recovery.json'),
              'billing_origin':{k:json.loads((state/'billing.json').read_text())[k] for k in ['start_date','usage0_cny']}}
    (suite/'resilient_policy.json').write_text(json.dumps(policy))
    policy_sha=base.file_hash(suite/'resilient_policy.json')
    helper_sha=policy['source_sha256']['retry_resilient_judgments.py']
    for tag in base.TAGS:
        for uid in run.questions:
            for order in (0,1):
                p=run.game_path(tag,uid,order)
                old=json.loads(p.read_text()) if p.exists() else None
                if old is not None and old['status']=='valid':continue
                request=run.request(tag,uid,order)
                row={'tag':tag,'uid':uid,'order':order,'attempt':0 if old is None else 1,
                     'local_request_id':f'offline-resilient-{tag}-{uid}-{order}',
                     'request':request,'request_sha256':judge.digest(request),
                     'baseline_model':base.BASELINE,'protocol_sha256':judge.digest(run.protocol),
                     'started_at':begin,'finished_at':end,'status':'valid','answer':'[[A=B]]',
                     'score':'A=B','finish_reason':'stop',
                     'usage':{'prompt_tokens':3,'completion_tokens':4,'total_tokens':7}}
                if old is not None:
                    archive=state/'attempts'/tag/f'{uid}-{order}'/(old['local_request_id']+'.json')
                    archive.parent.mkdir(parents=True,exist_ok=True)
                    archive.write_bytes(p.read_bytes())
                    row.update(supersedes_local_request_id=old['local_request_id'],
                               resilient_policy_sha256=policy_sha,resilient_helper_sha256=helper_sha)
                p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(row))
    run=controller.load_judge_run(suite);run.export()
    transport_run = 'offline-transport-run'
    transport_binding = {'schema':'arena_hard_connection_transport_v1','run_id':transport_run,
        'created_at':declared.isoformat(),'resilient_policy_sha256':policy_sha,
        'transport_source_sha256':policy['source_sha256']['resilient_judge.py'],
        'frozen_judge_sha256':policy['source_sha256']['source/scripts/judge_arena_hard.py'],
        'transport_max_connect_attempts':4,'backoff_seconds':[1.,2.,4.],
        'retry_exception_types':['httpx.ConnectError','httpx.ConnectTimeout'],
        'eligible_failure_phases':sorted(['connection.connect_tcp.failed','connection.start_tls.failed','proxy.start_tls.failed']),
        'require_no_application_headers_started':True,'max_connections':32,'max_keepalive_connections':32,
        'keepalive_expiry_seconds':300,'endpoint':'https://api.linkapi.ai/v1'}
    (suite/'job/transport_bindings').mkdir(parents=True)
    (suite/'job/transport_bindings'/f'{transport_run}.json').write_text(json.dumps(transport_binding))
    ledger=[]
    for p in (state/'games').glob('*/*.json'):
        row=json.loads(p.read_text())
        if str(p.relative_to(suite)) in preserved:continue
        common={'schema':'arena_hard_connection_transport_v1','run_id':transport_run,
                'audit_id':row['local_request_id'],'method':'POST','transport_attempt':1,
                'request_sha256':row['request_sha256']}
        ledger.append({**common,'event':'attempt_started','timestamp':begin})
        ledger.append({**common,'event':'attempt_finished','timestamp':end,'outcome':'response','status_code':200,
                       'elapsed_seconds':1.,'events':[{'event':'http11.send_request_headers.started','method':'POST'}],
                       'application_headers_started':True,'connection_failure_phase':None,
                       'retry_eligible':False,'will_retry':False,'backoff_seconds':0})
    (suite/'job/connect_attempts.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in ledger))
    decision={**controller.calculate_cost_decision(.65),
              'pilot_records_sha256':controller.pilot_status(suite,strict_scope=False)['records_sha256'],
              'resilient_policy_sha256':policy_sha,'decided_at':end}
    (suite/'resilient_cost_decision.json').write_text(json.dumps(decision))
    identity={k:policy[k] for k in ['source_sha256','original_experiment_sha256','legacy_manual_approval_sha256',
                                  'billing_origin','original_host_execution_sha256']}
    identity['resilient_policy_sha256']=policy_sha
    (suite/'resilient_execution.json').write_text(json.dumps({'identity':identity,
        'cost_decision_sha256':base.file_hash(suite/'resilient_cost_decision.json')}))
    scorer=load('complete_resilient_scorer',suite/'source/scripts/score_arena_hard.py')
    scorer.main(['--questions',str(suite/'question.jsonl'),'--answers-dir',str(suite/'model_answer'),
                 '--judgments-dir',str(run.directory),'--output',str(suite/'scores'),
                 '--baseline-model',base.BASELINE])
    return suite,api,base,policy


def test_real_six_thousand_requests_and_42_official_numeric_checks(complete_suite):
    suite,api,base,policy=complete_suite
    result=api.verify(suite)
    assert result['current_games']==result['exact_request_and_parse_checks']==6000
    assert result['numeric_comparisons']==42 and result['max_absolute_difference']<=1e-10
    assert result['total_recorded_api_attempts']==6003
    assert result['manual_transport_retries']==1
    assert result['initial_ambiguous_games']==3 and result['initial_invalid_games']==0
    assert result['resilient_retry_checks']['manual']==1
    assert result['resilient_retry_checks']['resilient']==2
    assert result['resilient_retry_checks']['ordinary']==5997
    assert result['transport_audit']['logical_requests_after_policy']==5978
    assert result['total_recorded_transport_attempts']==6003
    for name,expected in policy['preserved_valid_records_sha256'].items():
        assert base.file_hash(suite/name)==expected


@pytest.mark.parametrize('damage',['policy_scope','policy_max','transport_max','allowed_error',
    'original_host','old_manual_approval','source','preserved_valid','billing_origin',
    'execution_identity','cost_formula','cost_policy','pilot_hash'])
def test_final_binding_rejects_changed_policy_evidence_or_cost(complete_suite,damage):
    suite,api,base,policy=complete_suite
    targets={'policy_scope':'resilient_policy.json','policy_max':'resilient_policy.json',
      'transport_max':'resilient_policy.json','allowed_error':'resilient_policy.json',
      'original_host':'host_execution.json','old_manual_approval':'manual_transport_recovery.json',
      'source':'resilient_judge.py','preserved_valid':next(iter(policy['preserved_valid_records_sha256'])),
      'billing_origin':'model_judgment/gpt-4.1/state/billing.json','execution_identity':'resilient_execution.json',
      'cost_formula':'resilient_cost_decision.json','cost_policy':'resilient_cost_decision.json',
      'pilot_hash':'resilient_cost_decision.json'}
    path=suite/targets[damage];original=path.read_bytes()
    try:
        if damage in ['source','original_host','old_manual_approval','preserved_valid']:
            path.write_bytes(original+b'\n')
        else:
            d=json.loads(original)
            if damage=='policy_scope':d['policy']='unbounded'
            elif damage=='policy_max':d['max_total_attempts_per_game']=6
            elif damage=='transport_max':d['transport_max_connect_attempts']=10
            elif damage=='allowed_error':d['allowed_transport_errors'].append('ReadTimeout')
            elif damage=='billing_origin':d['usage0_cny']+=1
            elif damage=='execution_identity':d['identity']['original_experiment_sha256']='x'*64
            elif damage=='cost_formula':d['dispatch_budget_cny']+=1
            elif damage=='cost_policy':d['resilient_policy_sha256']='x'*64
            elif damage=='pilot_hash':d['pilot_records_sha256']='x'*64
            path.write_text(json.dumps(d))
        with pytest.raises(ValueError):api.verify(suite)
    finally:path.write_bytes(original)


def transport_case(tmp_path, phase='connection.connect_tcp.failed'):
    f=fixture_chain(tmp_path,(),legacy=False)
    api,base,judge,request,protocol,binding,state,rows,_=f
    record=rows[-1]
    record['finished_at']='2026-09-17T12:00:20Z'
    record_path=state/'games/sft-init/u1-0.json';record_path.parent.mkdir(parents=True)
    record_path.write_text(json.dumps(record))
    sources={'resilient_judge.py':'t'*64,'source/scripts/judge_arena_hard.py':'j'*64}
    binding['policy']['source_sha256']=sources
    auditdir=tmp_path/'job/transport_bindings';auditdir.mkdir(parents=True)
    transport={'schema':'arena_hard_connection_transport_v1','run_id':'run1','created_at':'2026-09-17T12:00:00Z',
        'resilient_policy_sha256':binding['policy_sha256'],'transport_source_sha256':'t'*64,
        'frozen_judge_sha256':'j'*64,'transport_max_connect_attempts':4,'backoff_seconds':[1.,2.,4.],
        'retry_exception_types':['httpx.ConnectError','httpx.ConnectTimeout'],
        'eligible_failure_phases':sorted(['connection.connect_tcp.failed','connection.start_tls.failed','proxy.start_tls.failed']),
        'require_no_application_headers_started':True,'max_connections':32,'max_keepalive_connections':32,
        'keepalive_expiry_seconds':300,'endpoint':'https://api.linkapi.ai/v1'}
    (auditdir/'run1.json').write_text(json.dumps(transport))
    common={'schema':'arena_hard_connection_transport_v1','run_id':'run1','audit_id':'audit1',
            'request_sha256':base.digest(request),'method':'POST'}
    prefix=[{'event':'http11.send_request_headers.started','method':'CONNECT'}] if phase.startswith('proxy.') else []
    events=[{**common,'transport_attempt':1,'event':'attempt_started','timestamp':'2026-09-17T12:00:01Z'},
            {**common,'transport_attempt':1,'event':'attempt_finished','timestamp':'2026-09-17T12:00:02Z',
             'outcome':'exception','exception_classes':['httpx.ConnectError','httpcore.ConnectError'],
             'raised_error_type':'ConnectError','events':prefix+[{'event':phase}],
             'application_headers_started':False,'connection_failure_phase':phase,
             'retry_eligible':True,'will_retry':True,'backoff_seconds':1.},
            {**common,'transport_attempt':2,'event':'attempt_started','timestamp':'2026-09-17T12:00:03Z'},
            {**common,'transport_attempt':2,'event':'attempt_finished','timestamp':'2026-09-17T12:00:04Z',
             'outcome':'response','status_code':200,
             'events':[{'event':'http11.send_request_headers.started','method':'POST'}],
             'application_headers_started':True,'connection_failure_phase':None,
             'retry_eligible':False,'will_retry':False,'backoff_seconds':0}]
    path=tmp_path/'job/connect_attempts.jsonl'
    path.write_text(''.join(json.dumps(r)+'\n' for r in events))
    return api,base,binding,path,events,record_path,auditdir/'run1.json'


@pytest.mark.parametrize('phase',['connection.connect_tcp.failed','connection.start_tls.failed','proxy.start_tls.failed'])
def test_transport_audit_counts_only_proven_connection_establishment_retries(tmp_path,phase):
    api,base,binding,*_=transport_case(tmp_path,phase)
    result=api.verify_transport_audit(base,binding)
    assert result['logical_requests_after_policy']==1
    assert result['transport_attempts_after_policy']==2
    assert result['connection_establishment_retries']==1


@pytest.mark.parametrize('damage',['missing_start','missing_finish','duplicate_start','ordinal','request','unknown_run',
    'get_in_paid_ledger','before_declared','read_timeout','no_connection_trace','headers_sent',
    'response_then_retry','unfinished_retry','five_attempts','wrong_record_interval','missing_logical',
    'binding_policy','binding_source','binding_limit','binding_pool','binding_endpoint'])
def test_transport_audit_rejects_unproven_or_unbound_attempts(tmp_path,damage):
    api,base,binding,path,events,record_path,bindpath=transport_case(tmp_path)
    if damage=='missing_start':events.pop(0)
    elif damage=='missing_finish':events.pop()
    elif damage=='duplicate_start':events[1]=copy.deepcopy(events[0])
    elif damage=='ordinal':events[2]['transport_attempt']=3
    elif damage=='request':events[1]['request_sha256']='x'*64
    elif damage=='unknown_run':events[0]['run_id']='other'
    elif damage=='get_in_paid_ledger':events[0]['method']='GET'
    elif damage=='before_declared':events[0]['timestamp']='2026-09-17T11:59:00Z'
    elif damage=='read_timeout':
        events[1].update(exception_classes=['httpx.ReadTimeout'],raised_error_type='ReadTimeout')
    elif damage=='no_connection_trace':events[1].update(events=[],connection_failure_phase=None)
    elif damage=='headers_sent':
        events[1]['events'].insert(0,{'event':'http11.send_request_headers.started','method':'POST'})
        events[1]['application_headers_started']=True
    elif damage=='response_then_retry':
        events[1].update(outcome='response',status_code=200)
    elif damage=='unfinished_retry':events=events[:2]
    elif damage=='five_attempts':events.extend(copy.deepcopy(events[:2])*3)
    elif damage=='wrong_record_interval':
        r=json.loads(record_path.read_text());r['started_at']='2026-09-17T12:00:10Z';record_path.write_text(json.dumps(r))
    elif damage=='missing_logical':
        other=record_path.with_name('u2-0.json');other.write_bytes(record_path.read_bytes())
    else:
        r=json.loads(bindpath.read_text())
        if damage=='binding_policy':r['resilient_policy_sha256']='x'*64
        elif damage=='binding_source':r['transport_source_sha256']='x'*64
        elif damage=='binding_limit':r['transport_max_connect_attempts']=5
        elif damage=='binding_pool':r['max_connections']=100
        elif damage=='binding_endpoint':r['endpoint']='https://other'
        bindpath.write_text(json.dumps(r))
    path.write_text(''.join(json.dumps(r)+'\n' for r in events))
    with pytest.raises(ValueError):api.verify_transport_audit(base,binding)
