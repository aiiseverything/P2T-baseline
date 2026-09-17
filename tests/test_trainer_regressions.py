"""Regression coverage for actual trainer control flow, using small CPU models."""
import json
import types
from collections import deque
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from torch import nn

from vpo_rm.trainer import TrainerConfig, VPOTrainer
from bytelevel_fixtures import ByteLevelTestTokenizer


@pytest.fixture
def isolated_benchmarks(tmp_path, monkeypatch):
    from vpo_rm import data

    benchmark = tmp_path / "benchmark.jsonl"
    benchmark.write_text(json.dumps({"instruction": "held-out benchmark prompt"}) + "\n")
    monkeypatch.setattr(data, "DEFAULT_BENCHMARK_PATHS", {"synthetic": benchmark})


class TinyTokenizer(ByteLevelTestTokenizer):
    def __init__(self):
        super().__init__({"<|endoftext|>": 0, "p": 1, "a": 2, "b": 3,
                          "c": 4, "d": 5, "<|im_end|>": 6})


class TinyActor(nn.Module):
    def __init__(self, rows=()):
        super().__init__()
        self.logits = nn.Parameter(torch.tensor([.4, .1, -.2, .3, .2, -.1, .1, 8.]))
        self.rows = deque(rows)
        self.generation_calls = []
        self.config = types.SimpleNamespace(attention_dropout=0., pad_token_id=0)
    def get_output_embeddings(self):
        return types.SimpleNamespace(weight=self.logits[:, None])
    def forward(self, input_ids, attention_mask, position_ids=None, logits_to_keep=None, **kwargs):
        width = len(logits_to_keep) if logits_to_keep is not None else input_ids.shape[1]
        return types.SimpleNamespace(logits=self.logits[None,None,:].expand(input_ids.shape[0], width, -1))
    def generate(self, input_ids, **kwargs):
        self.generation_calls.append(kwargs)
        return torch.cat((input_ids, torch.tensor([self.rows.popleft()])), -1)


def trainer(tmp_path, *, rows=(), **kwargs):
    cfg = TrainerConfig(actor_device="cpu", reward_device="cpu", output_dir=str(tmp_path),
                         group_size=2, prompts_per_rollout=1, min_response_tokens=1,
                         max_response_tokens=5, lora=False, gradient_checkpointing=False,
                         method="grpo", **kwargs)
    t = VPOTrainer(TinyActor(rows), TinyTokenizer(), nn.Linear(1,1), TinyTokenizer(), cfg)
    t.optimizer = torch.optim.SGD(t.actor.parameters(), lr=.05)
    t._encode_prompts = lambda prompts: ({"input_ids": torch.ones(len(prompts), 1, dtype=torch.long),
                                         "attention_mask": torch.ones(len(prompts), 1, dtype=torch.long)}, list(prompts))
    t._reward_batch = lambda *args: (torch.tensor([0.,1.]), None, None, None)
    return t


def fixed_rollout(t, responses=None, reasons=None):
    responses = torch.tensor([[2,3,0],[4,5,0]]) if responses is None else responses
    ids = torch.cat((torch.ones(len(responses),1,dtype=torch.long),responses),-1)
    masks = torch.ones_like(ids)
    positions = torch.arange(1,ids.shape[1])[None,:].expand(len(responses),-1)
    valid = torch.ones_like(responses,dtype=torch.bool)
    def rollout(prompts):
        t.actor.eval()
        values = (ids,masks,positions,responses,valid,["p"]*len(responses))
        # Preserve old API for red; fixed sampler adds explicit finish metadata.
        return values + (reasons or ["stop"]*len(responses),)
    t.rollout = rollout


def test_native_mask_keeps_first_mixed_stop_and_excludes_all_padding(tmp_path):
    t=trainer(tmp_path,rows=[[2,6],[3,4,5,0]])
    result=t.rollout(["p"])
    assert result[4].sum(-1).tolist()==[2,4]
    assert result[6]==["stop","stop"]


def test_native_sampler_receives_exact_training_support(tmp_path):
    t=trainer(tmp_path,rows=[[2,0],[3,0]])
    t.rollout(["p"])
    assert 7 in t.actor.generation_calls[0].get("suppress_tokens",[])


def test_native_finish_metadata_distinguishes_stop_on_cap_from_length(tmp_path):
    t=trainer(tmp_path,rows=[[2,3,4,5,0],[2,3,4,5,2]])
    result=t.rollout(["p"])
    assert len(result)==7
    assert result[6]==["stop","length"]


def test_requested_epochs_and_partial_optimizer_minibatches_are_executed(tmp_path):
    t=trainer(tmp_path,rows=[[2,0],[3,0]],policy_epochs_per_rollout=3,optimizer_minibatch_responses=1)
    with patch.object(t.optimizer,"step",wraps=t.optimizer.step) as step:
        t.train_rollout(["p"])
    assert step.call_count==6


@pytest.mark.parametrize("allocated_gpu_count",[0,2,3])
def test_gpu_hours_count_reserved_devices_since_trainer_start(tmp_path,allocated_gpu_count):
    t=trainer(tmp_path,rows=[[2,0],[3,0]],allocated_gpu_count=allocated_gpu_count)
    t._started=0.
    with patch("vpo_rm.trainer.time.monotonic",return_value=3600.):
        metrics=t.train_rollout(["p"])
    assert metrics["gpu_hours"]==float(allocated_gpu_count)


def test_initial_reference_regularizes_after_policy_drift(tmp_path):
    a=trainer(tmp_path/"a",rows=[[2,0],[3,0]],beta=0.)
    b=trainer(tmp_path/"b",rows=[[2,0],[3,0]],beta=2.)
    for t in [a,b]:
        with torch.no_grad(): t.actor.logits[2].add_(1.)
    a.train_rollout(["p"]); result=b.train_rollout(["p"])
    assert not torch.equal(a.actor.logits,b.actor.logits)
    assert result["kl_to_init"]>0


def test_nonfinite_gradient_never_reaches_optimizer_step(tmp_path):
    t=trainer(tmp_path,rows=[[2,0],[3,0]])
    t.actor.logits.register_hook(lambda grad: grad*float("nan"))
    before=t.actor.logits.detach().clone()
    with patch.object(t.optimizer,"step",wraps=t.optimizer.step) as step:
        with pytest.raises((RuntimeError,ValueError),match="[Nn]on.?finite"):
            t.train_rollout(["p"])
    assert step.call_count==0
    assert torch.equal(before,t.actor.logits)


def test_unsupported_sample_is_rejected_before_optimizer_step(tmp_path):
    t=trainer(tmp_path,rows=[[7,0],[3,0]])
    with patch.object(t.optimizer,"step",wraps=t.optimizer.step) as step:
        with pytest.raises(ValueError,match="support"):
            t.train_rollout(["p"])
    assert step.call_count==0


def test_reward_guard_uses_finish_reason_not_length(tmp_path):
    t=trainer(tmp_path,length_penalty_slope=.1,length_penalty_anchor=1)
    fixed_rollout(t,torch.tensor([[2,3,4,5,0],[2,3,4,5,2]]),["stop","length"])
    result=t.train_rollout(["p"])
    assert result["truncated_responses"]==1
    assert result["degenerate_responses"]==1


def test_sampling_temperature_and_minimum_stop_support_match_old_logp(tmp_path):
    t=trainer(tmp_path,temperature=2.)
    ids=torch.tensor([[1,2,0]])
    response=torch.tensor([[2,0]])
    valid=torch.ones_like(response,dtype=torch.bool)
    got=t._old_logp_microbatch(ids,torch.ones_like(ids),torch.tensor([[1,2]]),response,valid)
    scores=t.actor.logits.detach().double()/2
    # First token cannot stop; token 7 is never in tokenizer support.
    first=scores[2]-scores[torch.tensor([1,2,3,4,5])].logsumexp(0)
    second=scores[0]-scores[:7].logsumexp(0)
    torch.testing.assert_close(got,torch.tensor([[first,second]],dtype=torch.float32))


def test_bf16_policy_temperature_is_applied_after_float32_promotion(tmp_path):
    t=trainer(tmp_path,temperature=.7)
    t.actor.to(dtype=torch.bfloat16)
    ids=torch.tensor([[1,2,0]])
    response=torch.tensor([[2,0]])
    valid=torch.ones_like(response,dtype=torch.bool)
    got=t._old_logp_microbatch(ids,torch.ones_like(ids),torch.tensor([[1,2]]),response,valid)
    scores=t.actor.logits.detach().float()/.7
    expected=torch.stack((scores[2]-scores[torch.tensor([1,2,3,4,5])].logsumexp(0),
                          scores[0]-scores[:7].logsumexp(0)))[None,:]
    torch.testing.assert_close(got,expected,atol=1e-6,rtol=1e-6)


def test_zero_minimum_is_supported_without_empty_response(tmp_path):
    cfg=TrainerConfig(min_response_tokens=0).resolved()
    assert cfg.min_response_tokens==0
    t=trainer(tmp_path,rows=[[0],[6]])
    t.cfg.min_response_tokens=0
    metrics=t.train_rollout(["p"])
    assert metrics["response_tokens"]==2


@pytest.mark.parametrize("kwargs",[{"top_p":.8},{"top_k":2},{"lora_dropout":.1}])
def test_unsupported_sampling_or_dropout_is_rejected(kwargs):
    with pytest.raises(ValueError,match="support|dropout"):
        TrainerConfig(**kwargs).resolved()


def test_split_small_custom_file_never_reuses_validation_for_training(tmp_path, isolated_benchmarks):
    from scripts import train_skywork
    source=tmp_path/"prompts.txt"; source.write_text("first\nsecond\n")
    fake=types.SimpleNamespace(filter_prompts=lambda p:p,filtered_prompt_count=0,
                               train=lambda p:None,rollout_index=1,total_tokens=1)
    with patch.object(train_skywork.VPOTrainer,"from_pretrained",return_value=fake):
        train_skywork.main(["--prompts-file",str(source),"--output-dir",str(tmp_path/"out"),"--smoke"])
    meta=json.loads((tmp_path/"out"/"data_split.json").read_text())
    assert meta["validation_size"]==1
    assert meta["filtered_train_prompts"]==1


def test_existing_native_output_is_rejected_before_split_or_model_overwrite(tmp_path, isolated_benchmarks):
    from scripts import train_skywork
    source=tmp_path/"prompts.txt"; source.write_text("first\nsecond\n")
    out=tmp_path/"old"; out.mkdir()
    (out/"metrics.jsonl").write_text('{"rollout":1}\n')
    (out/"data_split.json").write_text('{"original":true}')
    fake=types.SimpleNamespace(filter_prompts=lambda p:p,filtered_prompt_count=0,
                               train=lambda p:None,rollout_index=1,total_tokens=1)
    with patch.object(train_skywork.VPOTrainer,"from_pretrained",return_value=fake):
        with pytest.raises(FileExistsError,match="fresh output"):
            train_skywork.main(["--prompts-file",str(source),"--output-dir",str(out),"--smoke"])
    assert (out/"data_split.json").read_text()=='{"original":true}'


def test_factory_rejects_existing_checkpoint_before_loading_model(tmp_path):
    (tmp_path/"checkpoint-1").mkdir()
    cfg=TrainerConfig(output_dir=str(tmp_path))
    with patch("transformers.AutoTokenizer.from_pretrained",side_effect=AssertionError("model loading began")):
        with pytest.raises(FileExistsError,match="fresh output"):
            VPOTrainer.from_pretrained(cfg)


def test_factory_seeds_before_lora_construction(tmp_path):
    import sys
    token=TinyTokenizer()
    class Base(TinyActor):
        def get_input_embeddings(self): return self.get_output_embeddings()
    class RM(nn.Module):
        def __init__(self):
            super().__init__(); self.base_model=Base(); self.score=nn.Linear(1,1)
            self.config=types.SimpleNamespace(pad_token_id=0)
    transformer=types.ModuleType("transformers")
    transformer.AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *a,**k:token)
    transformer.AutoModelForCausalLM=types.SimpleNamespace(from_pretrained=lambda *a,**k:Base())
    transformer.AutoModelForSequenceClassification=types.SimpleNamespace(from_pretrained=lambda *a,**k:RM())
    peft=types.ModuleType("peft")
    peft.LoraConfig=lambda **kw:kw
    peft.PeftModel=object
    def initialize(actor,config):
        actor.lora_A=nn.Parameter(torch.rand(4)); return actor
    peft.get_peft_model=initialize
    cfg=TrainerConfig(actor_device="cpu",reward_device="cpu",output_dir=str(tmp_path),gradient_checkpointing=False)
    values=[]
    with patch.dict(sys.modules,{"transformers":transformer,"peft":peft}),patch("vpo_rm.trainer.check_tokenizers"):
        for prior_seed in [1,2]:
            torch.manual_seed(prior_seed)
            values.append(VPOTrainer.from_pretrained(cfg).actor.lora_A.detach().clone())
    assert torch.equal(*values)


def test_once_sampler_uses_shared_protocol_and_preserves_finish_metadata(tmp_path, monkeypatch):
    import sys
    from scripts import vllm_generate_once
    captured = {}
    class Engine:
        def __init__(self, **kwargs): captured["engine"] = kwargs
        def generate(self, prompts, params, lora_request):
            captured["params"] = vars(params)
            return [types.SimpleNamespace(outputs=[
                types.SimpleNamespace(token_ids=[2,3,4,5,6],finish_reason="length",stop_reason=6),
                types.SimpleNamespace(token_ids=[2,3,4,5,2],finish_reason="length",stop_reason=None)])]
    module=types.ModuleType("vllm"); module.LLM=Engine
    module.SamplingParams=lambda **kwargs:types.SimpleNamespace(**kwargs)
    lora=types.ModuleType("vllm.lora.request"); lora.LoRARequest=lambda *args:args
    monkeypatch.setitem(sys.modules,"vllm",module)
    monkeypatch.setitem(sys.modules,"vllm.lora.request",lora)
    prompts=tmp_path/"prompts.json"; prompts.write_text('["p"]')
    output=tmp_path/"out.json"
    monkeypatch.setattr(sys,"argv",["generate","--model","fake","--adapter","fake",
        "--prompts",str(prompts),"--output",str(output),"--max-tokens","5",
        "--group-size","2","--temperature","2","--min-tokens","1","--seed","7"])
    with patch("transformers.AutoTokenizer.from_pretrained",return_value=TinyTokenizer()), \
         patch("transformers.AutoConfig.from_pretrained",return_value=types.SimpleNamespace(vocab_size=8)):
        vllm_generate_once.main()
    payload=json.loads(output.read_text())
    assert payload["finish_reasons"]==["stop","length"]
    assert payload["engine_finish_reasons"]==["length","length"]
    assert captured["params"]["logit_bias"]=={7:float("-inf")}
    assert captured["params"]["stop_token_ids"]==[0,6]
    assert captured["params"]["min_tokens"]==1
    assert captured["params"]["temperature"]==2.
    assert captured["engine"]["seed"]==7


def test_vllm_protocol_rejects_presence_penalty():
    from vpo_rm.integration import vllm_sampling_kwargs
    with pytest.raises(ValueError,match="presence_penalty"):
        vllm_sampling_kwargs(TinyTokenizer(),8,{"max_tokens":32,"presence_penalty":.3})


def test_large_vocabulary_uses_exact_compact_ban_instead_of_large_allowlist():
    from vpo_rm.integration import vllm_sampling_kwargs
    class LargeTokenizer(TinyTokenizer):
        def get_vocab(self): return {str(i):i for i in range(150000)}
    kwargs=vllm_sampling_kwargs(LargeTokenizer(),150267,{"max_tokens":32})
    assert "allowed_token_ids" not in kwargs
    assert len(kwargs["logit_bias"])==267
    assert all(value==float("-inf") for value in kwargs["logit_bias"].values())


def test_vllm_constructor_clamping_is_detected_and_summary_is_json_finite():
    from vpo_rm.integration import checked_sampling_params,sampling_summary
    def clamps(**kwargs):
        return types.SimpleNamespace(logit_bias={i:max(-100,value) for i,value in kwargs["logit_bias"].items()})
    kwargs={"logit_bias":{7:float("-inf")},"temperature":1.}
    with pytest.raises(RuntimeError,match="support"):
        checked_sampling_params(clamps,**kwargs)
    summary=sampling_summary(kwargs)
    json.dumps(summary,allow_nan=False)
    assert summary["suppressed_token_count"]==1


@pytest.mark.parametrize("temperature",[.001,3.])
def test_training_temperature_rejects_vllm_clamping_range(temperature):
    with pytest.raises(ValueError,match="temperature"):
        TrainerConfig(temperature=temperature).resolved()


def test_partial_minibatch_update_is_independent_of_microbatch_partition(tmp_path):
    parameters=[]
    for micro in [1,2,3]:
        t=trainer(tmp_path/str(micro),rows=[[2,0],[3,0],[4,0],[5,0]],
                  policy_epochs_per_rollout=2,optimizer_minibatch_responses=3,
                  microbatch_responses=micro,beta=.2)
        t._reward_batch=lambda *args:(torch.tensor([0.,1.,1.,0.]),None,None,None)
        result=t.train_rollout(["p","q"])
        assert result["optimizer_steps"]==4
        parameters.append(t.actor.logits.detach().clone())
    torch.testing.assert_close(parameters[0],parameters[1])
    torch.testing.assert_close(parameters[0],parameters[2])


def test_real_transformers_generate_scores_equal_teacher_sampling_logp(tmp_path):
    from transformers import GPTNeoXConfig,GPTNeoXForCausalLM
    torch.manual_seed(123)
    actor=GPTNeoXForCausalLM(GPTNeoXConfig(vocab_size=8,hidden_size=16,intermediate_size=24,
        num_hidden_layers=1,num_attention_heads=2,max_position_embeddings=32,
        bos_token_id=1,eos_token_id=0,pad_token_id=0,attention_dropout=0.,hidden_dropout=0.))
    cfg=TrainerConfig(actor_device="cpu",reward_device="cpu",output_dir=str(tmp_path),
        lora=False,gradient_checkpointing=False,group_size=2,max_response_tokens=5,
        min_response_tokens=2,temperature=.7,method="grpo")
    t=VPOTrainer(actor,TinyTokenizer(),nn.Linear(1,1),TinyTokenizer(),cfg)
    t._encode_prompts=lambda prompts:({"input_ids":torch.tensor([[1]]),"attention_mask":torch.ones(1,1,dtype=torch.long)},["p"])
    native_generate=actor.generate
    generated_logps=[]
    def capture(**kwargs):
        kwargs.update(return_dict_in_generate=True,output_scores=True)
        result=native_generate(**kwargs)
        tokens=result.sequences[:,1:]
        lp=torch.stack(result.scores,1).float().log_softmax(-1).gather(-1,tokens[...,None]).squeeze(-1)
        generated_logps.append(lp[0])
        return result.sequences
    with patch.object(actor,"generate",side_effect=capture):
        ids,mask,pos,responses,valid,_,_=t.rollout(["p"])
    old=t._old_logp_microbatch(ids,mask,pos,responses,valid)
    for i,expected in enumerate(generated_logps):
        torch.testing.assert_close(old[i,valid[i]],expected,atol=2e-6,rtol=2e-6)


def test_real_transformer_vpo_reward_gradient_and_optimizer_update(tmp_path):
    from transformers import GPTNeoXConfig,GPTNeoXForCausalLM
    from vpo_rm.reward import LastTokenReward
    class Backbone(nn.Module):
        def __init__(self):
            super().__init__(); self.emb=nn.Embedding(8,4)
            with torch.no_grad(): self.emb.weight.copy_(torch.arange(32).reshape(8,4)/32)
        def get_input_embeddings(self): return self.emb
        def forward(self,inputs_embeds,attention_mask,**kwargs):
            hidden=(inputs_embeds*attention_mask[...,None]).cumsum(1)
            return types.SimpleNamespace(last_hidden_state=hidden)
    torch.manual_seed(21)
    actor=GPTNeoXForCausalLM(GPTNeoXConfig(vocab_size=8,hidden_size=16,intermediate_size=24,
        num_hidden_layers=1,num_attention_heads=2,max_position_embeddings=32,
        attention_dropout=0.,hidden_dropout=0.))
    head=nn.Linear(4,1,bias=False)
    with torch.no_grad(): head.weight.fill_(1.)
    reward=LastTokenReward(Backbone(),head)
    cfg=TrainerConfig(actor_device="cpu",reward_device="cpu",output_dir=str(tmp_path),
        lora=False,gradient_checkpointing=False,group_size=2,max_response_tokens=5,
        min_response_tokens=1,method="vpo_rm",beta=.1,learning_rate=.01,
        policy_epochs_per_rollout=2,optimizer_minibatch_responses=1)
    t=VPOTrainer(actor,TinyTokenizer(),reward,TinyTokenizer(),cfg)
    fixed_rollout(t)
    before=actor.get_output_embeddings().weight.detach().clone()
    reference_before=t.reference_actor.get_output_embeddings().weight.detach().clone()
    metrics=t.train_rollout(["p"])
    assert metrics["optimizer_steps"]==4
    assert torch.isfinite(torch.tensor(metrics["loss"]))
    assert not torch.equal(before,actor.get_output_embeddings().weight)
    assert torch.equal(reference_before,t.reference_actor.get_output_embeddings().weight)
    assert all(p.grad is None for p in reward.parameters())


@pytest.mark.parametrize("sft_init",[False,True])
def test_real_peft_initial_reference_stays_frozen_across_updates(tmp_path,sft_init):
    peft=pytest.importorskip("peft")
    from transformers import GPTNeoXConfig,GPTNeoXForCausalLM
    torch.manual_seed(77)
    actor=GPTNeoXForCausalLM(GPTNeoXConfig(vocab_size=8,hidden_size=16,intermediate_size=24,
        num_hidden_layers=1,num_attention_heads=2,max_position_embeddings=32,
        attention_dropout=0.,hidden_dropout=0.))
    actor=peft.get_peft_model(actor,peft.LoraConfig(r=2,lora_alpha=4,lora_dropout=0,
        target_modules=["query_key_value"],task_type="CAUSAL_LM"))
    initial=""
    if sft_init:
        with torch.no_grad():
            for name,param in actor.named_parameters():
                if "lora_B.default" in name: param.fill_(.2)
        initial=str(tmp_path/"sft")
        actor.save_pretrained(initial)
        actor.load_adapter(initial,adapter_name="ref",is_trainable=False)
        actor.set_adapter("default")
    cfg=TrainerConfig(actor_device="cpu",reward_device="cpu",output_dir=str(tmp_path/"run"),
        lora=True,init_adapter=initial,gradient_checkpointing=False,group_size=2,
        max_response_tokens=5,min_response_tokens=1,method="grpo",beta=.5,
        learning_rate=.01,policy_epochs_per_rollout=2,optimizer_minibatch_responses=1)
    t=VPOTrainer(actor,TinyTokenizer(),nn.Linear(1,1),TinyTokenizer(),cfg)
    fixed_rollout(t)
    t._reward_batch=lambda *args:(torch.tensor([0.,1.]),None,None,None)
    ids,mask,pos,responses,valid,_,_=t.rollout(["p"])
    initial_logp=t._reference_logp(ids,mask,pos,responses,valid).clone()
    with torch.no_grad():
        for name,param in actor.named_parameters():
            if "lora_B.default" in name: param.add_(.3)
    current=t._old_logp_microbatch(ids,mask,pos,responses,valid)
    assert not torch.allclose(current,initial_logp)
    result=t.train_rollout(["p"])
    assert result["optimizer_steps"]==4
    torch.testing.assert_close(initial_logp,t._reference_logp(ids,mask,pos,responses,valid))
    assert actor.active_adapter=="default"
    assert all(param.grad is None for name,param in actor.named_parameters() if ".ref." in name)
