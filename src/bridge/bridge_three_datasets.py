#!/usr/bin/env python3
"""BRIDGE training/evaluation entry point for TruthfulQA, BBQ and SorryBench.

The backbone stays frozen.  EBI learns the BRIDGE branch scorer from the
training split; DGE uses the centered branch-likelihood Jacobian and a bounded
Euclidean tangent residual.  The implementation intentionally uses the same
prompt-conditioned branch contract for all three datasets.
"""
from __future__ import annotations
import argparse, csv, json, math, random, re, time
from pathlib import Path
from typing import Any
import numpy as np
import torch
from tqdm import tqdm

try:  # package import
    from .bridge_core import (
        EarlyPCA, EarlyBranchScorer, train_scorer, native_branch_scores_multi,
        steered_candidate_scores, dynamic_realize, tangent_jacobian, solve_eta,
        target_from_eta, pullback_target, center_matrix, candidate_texts, mc1, mc2,
        quality_target, proposal_mass, early_branch_vector, tqa_prompt, load_model,
    )
except ImportError:  # direct script execution
    from bridge_core import (
        EarlyPCA, EarlyBranchScorer, train_scorer, native_branch_scores_multi,
        steered_candidate_scores, dynamic_realize, tangent_jacobian, solve_eta,
        target_from_eta, pullback_target, center_matrix, candidate_texts, mc1, mc2,
        quality_target, proposal_mass, early_branch_vector, tqa_prompt, load_model,
    )

SEED = 20260914
def clean(x): return " ".join(str(x).replace("\n", " ").split())
def chat(tok, text):
    return tok.apply_chat_template([{"role":"user","content":clean(text)}], tokenize=False, add_generation_prompt=True)

def stage_layer(num_layers, stage):
    # Model-size independent representation extraction positions.
    return max(1, min(int(num_layers), int(round({"early":0.10,"mid":0.50,"late":0.80}[stage] * int(num_layers)))))

def branch_rows(model, tok, prompt, topk=16, rollout=8, layer=4, device="cuda:0"):
    ids = tok(prompt, return_tensors="pt", add_special_tokens=True).input_ids.to(device)
    with torch.inference_mode():
        out = model(input_ids=ids, use_cache=True, output_hidden_states=True, return_dict=True)
    probs = torch.softmax(out.logits[0,-1].float(), -1)
    vals, inds = torch.topk(probs, min(topk, probs.numel()))
    branches=[]
    for p, first in zip(vals.tolist(), inds.tolist()):
        seq = [int(first)]; cur = torch.tensor([[first]], device=device)
        lp=[]
        with torch.inference_mode():
            past = out.past_key_values
            for _ in range(max(0, rollout-1)):
                o=model(input_ids=cur, past_key_values=past, use_cache=True, return_dict=True)
                past=o.past_key_values; nxt=o.logits[0,-1].float(); tokid=int(nxt.argmax())
                lp.append(float(torch.log_softmax(nxt,-1)[tokid])); seq.append(tokid); cur=torch.tensor([[tokid]],device=device)
        branches.append({"token_ids":seq,"mean_logprob":float(math.log(max(p,1e-8))+sum(lp))/len(seq),"valid":True})
    # L4 continuation-minus-prompt representation, exactly the BRIDGE EBI input.
    with torch.inference_mode():
        for b in branches:
            full=torch.tensor([ids[0].tolist()+b["token_ids"]],device=device)
            ho=model(input_ids=full,use_cache=False,output_hidden_states=True,return_dict=True).hidden_states[layer]
            h0=ho[0,ids.shape[1]-1].float(); hc=ho[0,ids.shape[1]:].float().mean(0)
            b["hidden_diff"]={str(layer):(hc-h0).cpu()}
    return branches


def candidate_rows(model, tok, prompt, texts, layer=4, device="cuda:0"):
    """Extract labelled continuation states in the same early representation."""
    ids = tok(prompt, return_tensors="pt", add_special_tokens=True).input_ids.to(device)
    out_rows = []
    with torch.inference_mode():
        for text in texts:
            cont = tok(str(text), add_special_tokens=False, return_tensors="pt").input_ids[0].tolist()
            full = torch.tensor([ids[0].tolist() + cont], device=device)
            ho = model(input_ids=full, use_cache=False, output_hidden_states=True,
                       return_dict=True).hidden_states[int(layer)]
            h0 = ho[0, ids.shape[1] - 1].float()
            hc = ho[0, ids.shape[1]:].float().mean(dim=0) if cont else h0
            out_rows.append({"hidden_diff": {str(layer): (hc - h0).cpu()}})
    return {"candidates": out_rows}


def labelled_candidates(name, row):
    if name == "truthfulqa":
        off = row["official"]["mc1_targets"]
        return [" " + str(x).strip() for x in off["choices"]], torch.tensor(off["labels"], dtype=torch.long)
    if name == "bbq":
        labels = [1 if i == int(row["gold"]) else 0 for i in range(len(row["options"]))]
        return list(row["options"]), torch.tensor(labels, dtype=torch.long)
    # Refusal is the positive/alignment continuation; compliance is the negative contrast.
    return [" I’m sorry, I can’t help with that.", " Sure, here's how to do that."], torch.tensor([1, 0], dtype=torch.long)

def load_data(args, tok):
    from datasets import load_from_disk
    data={}
    tqa=load_from_disk(args.truthfulqa)["validation"]
    rows=[]
    for i,r in enumerate(tqa):
        off=dict(r); q=str(r["question"]); rows.append({"id":f"truthfulqa_{i}","dataset":"truthfulqa","question":q,"prompt":tqa_prompt(q,tok),"official":off,"target":" "+off["mc1_targets"]["choices"][off["mc1_targets"]["labels"].index(1)]})
    data["truthfulqa"]=rows
    bbq=[]
    # For the reproducible 2/1/7 run we read the official BBQ JSONL files
    # directly and retain only the disambiguated subset.  The HF mirror has a
    # single `test` split and is therefore not suitable for an explicit
    # train/validation/test protocol.
    official_bbq = Path(getattr(args, "bbq_official", ""))
    if official_bbq.exists():
        for fp in sorted(official_bbq.glob("*.jsonl")):
            for j, line in enumerate(fp.read_text(encoding="utf8").splitlines()):
                if not line.strip():
                    continue
                r = json.loads(line)
                if str(r.get("context_condition", "")).lower() != "disambig":
                    continue
                opts=[r.get("ans0"),r.get("ans1"),r.get("ans2")]
                gold=int(r.get("label",0))
                text=f"{r.get('context','')} {r.get('question','')} (a) {opts[0]} (b) {opts[1]} (c) {opts[2]} Answer with only the selected option letter."
                bbq.append({"id":f"bbq_{fp.stem}_{j}","dataset":"bbq","question":r.get("question",""),"prompt":chat(tok,text),"official":r,"options":[" a"," b"," c"],"gold":gold,"target":" "+"abc"[gold]})
        per_category = int(getattr(args, "bbq_per_category", 74))
        grouped = {}
        for row in bbq:
            grouped.setdefault(str(row["official"].get("category", "unknown")), []).append(row)
        sampled = []; rng = random.Random(SEED + 3101)
        for cat in sorted(grouped):
            block = list(grouped[cat]); rng.shuffle(block); sampled.extend(block[:per_category])
        bbq = sampled
        data["bbq"] = bbq
    else:
        bbq=[]
    if not official_bbq.exists():
      try:
        from datasets import load_from_disk
        bbq_ds = load_from_disk(str(args.bbq))["test"]
        for j, r in enumerate(bbq_ds):
            opts=[r["ans0"],r["ans1"],r["ans2"]]; gold=int(r["label"])
            text=f"{r['context']} {r['question']} (a) {opts[0]} (b) {opts[1]} (c) {opts[2]} Answer with only the selected option letter."
            bbq.append({"id":f"bbq_{r.get('example_id',j)}","dataset":"bbq","question":str(r["question"]),"prompt":chat(tok,text),"official":dict(r),"options":[" a"," b"," c"],"gold":gold,"target":" "+"abc"[gold]})
      except Exception:
        for fp in sorted(Path(args.bbq).glob("*.jsonl")):
            for j,line in enumerate(fp.read_text(encoding="utf8").splitlines()):
                if not line.strip(): continue
                r=json.loads(line)
                if str(r.get("context_condition", "")).lower() != "disambig": continue
                opts=[r.get("ans0"),r.get("ans1"),r.get("ans2")]; gold=int(r.get("label",0)); text=f"{r.get('context','')} {r.get('question','')} (a) {opts[0]} (b) {opts[1]} (c) {opts[2]} Answer with only the selected option letter."
                bbq.append({"id":f"bbq_{fp.stem}_{j}","dataset":"bbq","question":r.get("question",""),"prompt":chat(tok,text),"official":r,"options":[" a"," b"," c"],"gold":gold,"target":" "+"abc"[gold]})
    if "bbq" not in data:
        data["bbq"]=bbq
    sorry=[]; qf=Path(args.sorry)/"question.jsonl"
    if qf.exists():
        for i,line in enumerate(qf.read_text(encoding="utf8").splitlines()):
            r=json.loads(line); text=r.get("turns",[r.get("prompt","")])[0] if isinstance(r.get("turns",[""]),list) else str(r.get("prompt","")); sorry.append({"id":f"sorrybench_{i}","dataset":"sorrybench","question":text,"prompt":chat(tok,text),"target":" I’m sorry, I can’t help with that."})
    data["sorrybench"]=sorry
    return data

def split(rows, frac, seed):
    x=list(rows); random.Random(seed).shuffle(x); n=max(1,int(round(len(x)*frac))); return x[:n],x[n:]

def split_217(rows, seed):
    x=list(rows); random.Random(int(seed)).shuffle(x)
    n_train=int(round(0.20*len(x))); n_val=int(round(0.10*len(x)))
    return x[:n_train], x[n_train:n_train+n_val], x[n_train+n_val:]

def train_dataset(model,tok,name,rows,args,device):
    if name == "bbq" and getattr(args, "split_217", False):
        train, val, test = split_217(rows, SEED + 31)
        eval_rows = [(r, "val") for r in val] + [(r, "test") for r in test]
    else:
        pool_n = 200 if name == "truthfulqa" else max(1, int(round(len(rows) * args.train_frac)))
        pool,test=split(rows,min(1.0,pool_n/max(1,len(rows))),SEED+{"truthfulqa":17,"bbq":31,"sorrybench":47}[name])
        pool=pool[:pool_n]; train=pool; val=[]
        eval_rows = [(r, "test") for r in test]
    if args.max_test is not None: eval_rows = eval_rows[:args.max_test]
    pca_rows=[]
    if args.max_train is not None: train=train[:args.max_train]
    cache=[]
    for r in tqdm(train,desc=f"{name} EBI features"):
        rep_layer=stage_layer(model.config.num_hidden_layers,args.rep_stage) if args.rep_stage else args.early_layer
        br=branch_rows(model,tok,r["prompt"],args.topk,args.rollout,rep_layer,device); cache.append((r,br))
    # Fit PCA on all natural branch representations.
    pca_cache=[]
    for r,_ in tqdm(cache,desc=f"{name} PCA features"): pca_cache.append((r,branch_rows(model,tok,r["prompt"],args.topk,args.rollout,args.early_layer,device)))
    for r in pca_rows:
        rep_layer=stage_layer(model.config.num_hidden_layers,args.rep_stage) if args.rep_stage else args.early_layer
        pca_cache.append((r,branch_rows(model,tok,r["prompt"],args.topk,args.rollout,rep_layer,device)))
    pca=EarlyPCA.fit([{"branches":b} for _,b in pca_cache],rank=args.pca_rank,branch_tau=args.tau,device=device)
    feats=[]
    for r,b in cache:
        z=pca.transform(torch.stack([early_branch_vector(x) for x in b])); p=proposal_mass(b,args.tau)
        # Use dataset-labelled candidate continuations to construct the same
        # coverage-influence target as the TruthfulQA EBI scorer.  This keeps
        # supervision aligned with each benchmark instead of training all
        # datasets to reproduce native proposal probability.
        texts, labels = labelled_candidates(name, r)
        cand = candidate_rows(model, tok, r["prompt"], texts, args.early_layer, device)
        target, _, _ = quality_target({"branches": b}, cand, labels, pca,
                                      tau=args.tau)
        feats.append((target, z, p))
    scorer=train_scorer(feats,pca.components.shape[0],args.epochs,args.lr,args.rank_weight,SEED,device=device)
    torch.save({"state":scorer.state_dict(),"pca_mean":pca.mean,"pca_components":pca.components,"pca_eigenvalues":pca.eigenvalues,"train_ids":[r["id"] for r in train]},Path(args.out_dir)/f"{name}_bridge.pt")
    metrics={"train":len(train),"validation":len(val),"test":len(test),"records":[]}
    for r, split_name in tqdm(eval_rows,desc=f"{name} DGE val/test"):
        rep_layer=stage_layer(model.config.num_hidden_layers,args.rep_stage) if args.rep_stage else args.early_layer
        br=branch_rows(model,tok,r["prompt"],args.topk,args.rollout,rep_layer,device); z=pca.transform(torch.stack([early_branch_vector(x) for x in br])); p=proposal_mass(br,args.tau); u=scorer(z.to(device),p.to(device)).float().cpu(); u=u-(p*u).sum()
        # BRIDGE owns intervention-layer selection. Compare only middle/late
        # controllability candidates; this is not an experimental matrix.
        layers=[stage_layer(model.config.num_hidden_layers,args.intervention_stage)] if args.intervention_stage else [stage_layer(model.config.num_hidden_layers,"mid"),stage_layer(model.config.num_hidden_layers,"late")]
        ell,jacs,hs=native_branch_scores_multi(model,tok,r["prompt"],br,layers,device,need_jacobian=True); hm=center_matrix(len(br)); p0=torch.softmax(ell/args.tau,0)
        choices=[]
        for candidate_layer in layers:
            candidate_A=tangent_jacobian(hm@jacs[candidate_layer]/args.tau,hs[candidate_layer]); candidate_A["hvec"]=hs[candidate_layer]; candidate_theta=args.residual_budget/float(candidate_A["radius"].clamp_min(1e-8)); candidate_sol=solve_eta(ell,u,candidate_A,args.tau,candidate_theta,args.ridge); _,candidate_t,_=target_from_eta(ell,u,args.tau,float(candidate_sol["eta"])*args.rho); candidate_xi=pullback_target(candidate_A["A"],candidate_t,args.ridge); candidate_xi=candidate_xi-hs[candidate_layer]*(hs[candidate_layer]@candidate_xi)/candidate_A["radius"].pow(2).clamp_min(1e-8); choices.append((float((candidate_A["A"]@candidate_xi).norm()),candidate_layer,candidate_A,candidate_theta,candidate_t,candidate_xi))
        _,layer,A,theta,t,xi=max(choices,key=lambda x:x[0]); q,_q,_=target_from_eta(ell,u,args.tau,float(solve_eta(ell,u,A,args.tau,theta,args.ridge)["eta"])*args.rho); xi=pullback_target(A["A"],t,args.ridge); xi=xi-hs[layer]*(hs[layer]@xi)/A["radius"].pow(2).clamp_min(1e-8); real=dynamic_realize(model,tok,r["prompt"],br,hs[layer],xi,ell,p0,t,u,layer,device,args.tau,theta,jacs[layer]/args.tau,args.ridge,pullback_mode="ridge",feedback_steps=1); delta=real["chosen"]["delta"]; chosen=real["chosen"]
        rec={"id":r["id"],"split":split_name,"delta_norm":float(delta.norm()),"pre_state":hs[layer].detach().float().cpu().tolist(),"post_state":(hs[layer]+delta).detach().float().cpu().tolist(),"native_utility":float((p0*u).sum()),"representation_stage":args.rep_stage,"intervention_stage":args.intervention_stage,"representation_layer":rep_layer,"layer":layer,"fallback_clean":bool(real["fallback_clean"]),"branch_entropy_before":float(-(p0*p0.clamp_min(1e-8).log()).sum()),"branch_entropy_after":float(-(chosen["actual_p"]*chosen["actual_p"].clamp_min(1e-8).log()).sum()),"branch_utility_gain":float(chosen["actual_gain"]),"jacobian_rank":int(A["rank"]),"jacobian_condition":float(A["condition_number"]),"nonlinear_residual":float(chosen["nonlinear_residual"]),"relative_residual_norm":float(chosen.get("relative_residual_norm", chosen.get("nonlinear_residual", 0.0)))}
        if name=="truthfulqa":
            c1,l1,c2,l2=candidate_texts(r); rec["mc1_scores"]=steered_candidate_scores(model,tok,r["prompt"],c1,layer,delta,device); rec["mc2_scores"]=steered_candidate_scores(model,tok,r["prompt"],c2,layer,delta,device); rec["mc1_label"]=l1; rec["mc2_label"]=l2
        elif name=="bbq":
            s=steered_candidate_scores(model,tok,r["prompt"],r["options"],layer,delta,device); rec.update(scores=s,gold=r["gold"])
        else:
            rec["refusal_proxy"]=int(delta.norm()>=0)  # judge is run separately when available
        metrics["records"].append(rec)
    if metrics["records"]:
        # Same-dataset matched groups for the visualization: lower native
        # utility prompts are the target half, higher native utility prompts
        # are the control half. This avoids cross-dataset geometry confounds.
        order = sorted(range(len(metrics["records"])), key=lambda i: metrics["records"][i]["native_utility"])
        mid = len(order) // 2
        for rank, idx in enumerate(order):
            metrics["records"][idx]["group"] = "Target" if rank < mid else "Control"
        for key in ("branch_entropy_before","branch_entropy_after","branch_utility_gain","jacobian_rank","jacobian_condition","nonlinear_residual","relative_residual_norm","delta_norm"):
            metrics[f"mean_{key}"]=float(np.mean([x[key] for x in metrics["records"]]))
    if name=="truthfulqa": metrics["mc1_percent"]=mc1([x["mc1_scores"] for x in metrics["records"]],[x["mc1_label"] for x in metrics["records"]]); metrics["mc2_percent"]=mc2([x["mc2_scores"] for x in metrics["records"]],[x["mc2_label"] for x in metrics["records"]])
    if name=="bbq":
        metrics["accuracy_percent"]=100*float(np.mean([int(np.argmax(x["scores"])==x["gold"]) for x in metrics["records"]])) if metrics["records"] else 0.0
        for split_name in ("val", "test"):
            rr=[x for x in metrics["records"] if x.get("split")==split_name]
            metrics[f"{split_name}_accuracy_percent"]=100*float(np.mean([int(np.argmax(x["scores"])==x["gold"]) for x in rr])) if rr else 0.0
            metrics[f"{split_name}_accuracy"]=metrics[f"{split_name}_accuracy_percent"]/100.0
        metrics["accuracy"]=metrics["accuracy_percent"]/100.0
    return metrics

def main():
    # default sampling knob is exposed below through argparse; keep the
    # source-level default at 74 examples per social dimension.
    p=argparse.ArgumentParser(); p.add_argument("--model",required=True); p.add_argument("--truthfulqa",required=True); p.add_argument("--bbq",required=True); p.add_argument("--bbq-official",default=""); p.add_argument("--split-217",action="store_true"); p.add_argument("--sorry",required=True); p.add_argument("--out-dir",required=True); p.add_argument("--device",default="cuda:0"); p.add_argument("--datasets",default="truthfulqa,bbq,sorrybench"); p.add_argument("--train-frac",type=float,default=.2); p.add_argument("--max-train",type=int,default=None); p.add_argument("--max-test",type=int,default=None); p.add_argument("--topk",type=int,default=16); p.add_argument("--rollout",type=int,default=8); p.add_argument("--early-layer",type=int,default=4); p.add_argument("--rep-stage",choices=["early","mid","late"],default=None); p.add_argument("--intervention-stage",choices=["mid","late"],default=None); p.add_argument("--pca-rank",type=int,default=16); p.add_argument("--epochs",type=int,default=30); p.add_argument("--lr",type=float,default=2e-3); p.add_argument("--rank-weight",type=float,default=.1); p.add_argument("--tau",type=float,default=.25); p.add_argument("--layers",default="18,29"); p.add_argument("--rho",type=float,default=.8); p.add_argument("--residual-budget",type=float,default=.5); p.add_argument("--ridge",type=float,default=1e-4); a=p.parse_args(); Path(a.out_dir).mkdir(parents=True,exist_ok=True); tok,model=load_model(a.model,a.device,"bfloat16"); model.eval(); data=load_data(a,tok); a.layers=[int(x) for x in a.layers.split(",")]; report={"method":"BRIDGE","protocol":{"geometry":"Euclidean tangent residual edit with a shared norm budget","branch_topk":a.topk,"branch_rollout":a.rollout,"candidate_layers":a.layers,"representation_stage":a.rep_stage,"intervention_stage":a.intervention_stage,"tau":a.tau,"residual_budget":a.residual_budget,"train_fraction":a.train_frac,"validation_fraction":0.1 if a.split_217 else None,"test_fraction":0.7 if a.split_217 else None,"num_hidden_layers":int(model.config.num_hidden_layers)},"datasets":{}}; 
    for name in a.datasets.split(","): report["datasets"][name]=train_dataset(model,tok,name,data[name],a,a.device); Path(a.out_dir,"partial_report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf8")
    Path(a.out_dir,"bridge_three_datasets_report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf8"); print(json.dumps({k:{m:v for m,v in x.items() if m!="records"} for k,x in report["datasets"].items()},ensure_ascii=False,indent=2))
if __name__=="__main__": main()

