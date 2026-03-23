import common
import torch
import os
from typing import List
from language_models import GPT, HuggingFace
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import (
    VICUNA_PATH, VICUNA_13B_PATH, LLAMA_7B_PATH, LLAMA_13B_PATH, LLAMA_70B_PATH,
    LLAMA3_8B_PATH, LLAMA3_70B_PATH, LLAMA31_8B_PATH,
    GEMMA_2B_PATH, GEMMA_7B_PATH, MISTRAL_7B_PATH, MIXTRAL_7B_PATH,
    R2D2_PATH, PHI3_MINI_PATH, QWEN2_5_7B_PATH,
    TARGET_TEMP, TARGET_TOP_P,
)
import torch.nn.functional as F


def load_target_model(args):
    return TargetLM(
        model_name=args.target_model,
        temperature=TARGET_TEMP,
        top_p=TARGET_TOP_P,
        persona_prompt=args.persona_prompt,
        sys_mode=args.sys_mode,
        safe_prompt=getattr(args, "safe_prompt", ""),
    )


def _build_system_text(sys_mode, has_persona, safe_prompt, persona_prompt):
    """
    Construct the final system message from safe_prompt and persona_prompt.

    Default (no safe_prompt, no persona): ""
    With safe_prompt only:               safe_prompt
    With persona only (replace):         persona_prompt
    With both (append):                  safe_prompt + "\n\n" + persona_prompt
    With both (prepend):                 persona_prompt + "\n\n" + safe_prompt
    """
    if not has_persona:
        return safe_prompt

    if sys_mode == "replace":
        return persona_prompt
    if sys_mode == "append":
        return f"{safe_prompt}\n\n{persona_prompt}" if safe_prompt else persona_prompt
    if sys_mode == "prepend":
        return f"{persona_prompt}\n\n{safe_prompt}" if safe_prompt else persona_prompt
    return safe_prompt


class TargetLM:
    def __init__(self, model_name: str, temperature: float, top_p: float,
                 persona_prompt: str, sys_mode: str, safe_prompt: str = ""):
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.model, self.template = load_indiv_model(model_name)
        self.n_input_tokens = 0
        self.n_output_tokens = 0
        self.n_input_chars = 0
        self.n_output_chars = 0
        self.persona_prompt = persona_prompt
        self.sys_mode = sys_mode
        self.safe_prompt = safe_prompt

    def _build_full_prompts(self, prompts_list, persona_prompt, convs_list):
        tokenizer = self.model.tokenizer
        has_persona = bool(persona_prompt and persona_prompt.strip())
        system_text = _build_system_text(
            self.sys_mode, has_persona, self.safe_prompt, persona_prompt or ""
        )
        full_prompts = []

        for conv, prompt in zip(convs_list, prompts_list):
            if "mistral" in self.model_name or "gemma" in self.model_name:
                sys_prefix = "SYSTEM PROMPT: " if "mistral" in self.model_name else ""
                sys_suffix = "\n\n###\n\nUSER: " + prompt
                prompt = f"{sys_prefix}{system_text}{sys_suffix}" if system_text else prompt

            elif "llama2" in self.model_name:
                SYS_START = "<s>[INST] <<SYS>>\n"
                SYS_END = "\n<</SYS>>\n\n"
                conv.system = f"{SYS_START}{system_text}{SYS_END}" if system_text else ""
                prompt = prompt + " "

            elif any(m in self.model_name for m in ["llama3", "phi3", "vicuna", "qwen2.5", "r2d2"]):
                conv.system_template = "{system_message}"
                conv.system_message = system_text

            elif "gpt-4" in self.model_name or "gpt-3.5-turbo" in self.model_name:
                conv.system_template = "{system_message}"
                conv.system_message = system_text

            conv.append_message(conv.roles[0], prompt)

            if "gpt" in self.model_name:
                full_prompts.append(conv.to_openai_api_messages())
            elif "vicuna" in self.model_name:
                conv.append_message(conv.roles[1], None)
                full_prompts.append(conv.get_prompt())
            elif "llama2" in self.model_name:
                conv.append_message(conv.roles[1], None)
                full_prompts.append("<s>" + conv.get_prompt())
            elif any(m in self.model_name for m in ["r2d2", "gemma", "mistral", "llama3", "qwen2.5", "phi3"]):
                conv_list_dicts = conv.to_openai_api_messages()
                if "gemma" in self.model_name:
                    conv_list_dicts = conv_list_dicts[1:]
                full_prompt = tokenizer.apply_chat_template(
                    conv_list_dicts, tokenize=False, add_generation_prompt=True
                )
                full_prompts.append(full_prompt)
            else:
                raise ValueError(f"Unknown model: {self.model_name}")

        return full_prompts

    def get_response_final(self, prompts_list: List[str], persona_prompt=None,
                           max_n_tokens=None, temperature=None, no_template=False) -> List[dict]:
        batchsize = len(prompts_list)
        convs_list = [common.conv_template(self.template) for _ in range(batchsize)]

        if no_template:
            full_prompts = prompts_list
        else:
            full_prompts = self._build_full_prompts(prompts_list, persona_prompt, convs_list)

        current_temp = self.temperature if temperature is None else temperature
        if "gpt" not in self.model_name.lower():
            self.model.model.generation_config.temperature = current_temp
            self.model.model.generation_config.do_sample = current_temp > 0
            self.model.model.generation_config.top_p = self.top_p

        outputs = self.model.generate(
            full_prompts,
            max_n_tokens=max_n_tokens,
            temperature=current_temp,
            top_p=self.top_p,
        )

        self.n_input_tokens += sum(o["n_input_tokens"] for o in outputs)
        self.n_output_tokens += sum(o["n_output_tokens"] for o in outputs)
        self.n_input_chars += sum(len(p) for p in full_prompts)
        self.n_output_chars += sum(len(o["text"]) for o in outputs)
        return outputs

    def get_topk_first_only(self, prompts_list: List[str], persona_prompt: str = None,
                            topk: int = 20, no_template: bool = False) -> List[dict]:
        if "gpt" in self.model_name.lower():
            raise NotImplementedError("Top-k logits extraction is not supported for GPT API models.")

        batchsize = len(prompts_list)
        tokenizer = self.model.tokenizer
        convs_list = [common.conv_template(self.template) for _ in range(batchsize)]

        if no_template:
            full_prompts = prompts_list
        else:
            full_prompts = self._build_full_prompts(prompts_list, persona_prompt, convs_list)

        inputs = tokenizer(full_prompts, return_tensors="pt", padding=True, truncation=False)
        device = self.model.model.device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        out = self.model.model(**inputs)
        logits = out.logits

        last_pos = (
            inputs["attention_mask"].sum(dim=1) - 1
            if "attention_mask" in inputs
            else torch.full((logits.size(0),), logits.size(1) - 1, device=logits.device, dtype=torch.long)
        )

        batch_idx = torch.arange(logits.size(0), device=logits.device)
        next_logits = logits[batch_idx, last_pos, :]
        probs = F.softmax(next_logits, dim=-1)
        top_probs, top_ids = torch.topk(probs, k=topk, dim=-1)

        results = []
        for b in range(batchsize):
            items = []
            for r in range(topk):
                tid = int(top_ids[b, r].item())
                items.append({
                    "rank": r + 1,
                    "token_id": tid,
                    "token_str": tokenizer.convert_ids_to_tokens(tid),
                    "token_text": tokenizer.decode([tid], clean_up_tokenization_spaces=False),
                    "prob": float(top_probs[b, r].item()),
                    "logit": float(next_logits[b, tid].item()),
                })
            results.append({"topk_first": items})

        return results

    def get_topk_greedy_steps_llama2(self, prompts_list, persona_prompt=None, topk=20, n_steps=5):
        assert "llama2" in self.model_name.lower(), "This function is for LLaMA-2 only."

        tokenizer = self.model.tokenizer
        device = self.model.model.device
        batchsize = len(prompts_list)
        convs_list = [common.conv_template(self.template) for _ in range(batchsize)]
        has_persona = bool(persona_prompt and persona_prompt.strip())
        system_text = _build_system_text(
            self.sys_mode, has_persona, self.safe_prompt, persona_prompt or ""
        )

        full_prompts = []
        for conv, prompt in zip(convs_list, prompts_list):
            SYS_START = "<s>[INST] <<SYS>>\n"
            SYS_END = "\n<</SYS>>\n\n"
            conv.system = f"{SYS_START}{system_text}{SYS_END}" if system_text else ""
            prompt = prompt + " "
            conv.append_message(conv.roles[0], prompt)
            conv.append_message(conv.roles[1], None)
            full_prompts.append("<s>" + conv.get_prompt())

        inputs = tokenizer(full_prompts, return_tensors="pt", padding=True)
        cur_ids = inputs["input_ids"].to(device)
        cur_mask = inputs.get("attention_mask", None)
        if cur_mask is not None:
            cur_mask = cur_mask.to(device)

        all_results = []
        for t in range(1, n_steps + 1):
            with torch.no_grad():
                out = self.model.model(input_ids=cur_ids, attention_mask=cur_mask)
                next_logits = out.logits[:, -1, :]
                probs = F.softmax(next_logits, dim=-1)
                top_probs, top_ids = torch.topk(probs, k=topk, dim=-1)
                top_logits = torch.gather(next_logits, dim=-1, index=top_ids)

            step_payload = []
            for b in range(batchsize):
                arr = []
                for r in range(topk):
                    tid = int(top_ids[b, r].item())
                    tok_str = tokenizer.convert_ids_to_tokens(tid)
                    arr.append({
                        "step": t, "rank": r + 1, "token_id": tid,
                        "token_str": tok_str,
                        "token_text": tok_str.replace("▁", " ").strip(),
                        "prob": float(top_probs[b, r].item()),
                        "logit": float(top_logits[b, r].item()),
                    })
                step_payload.append(arr)

            if t == 1:
                all_results = [{"topk_steps": []} for _ in range(batchsize)]
            for b in range(batchsize):
                all_results[b]["topk_steps"].append(step_payload[b])

            greedy_ids = top_ids[:, 0].unsqueeze(1)
            cur_ids = torch.cat([cur_ids, greedy_ids], dim=1)
            if cur_mask is not None:
                cur_mask = torch.cat(
                    [cur_mask, torch.ones((batchsize, 1), device=device, dtype=cur_mask.dtype)], dim=1
                )

        return all_results


def load_indiv_model(model_name, device=None):
    model_path, template = get_model_path_and_template(model_name)

    if "gpt" in model_name or "together" in model_name:
        lm = GPT(model_name)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto",
            trust_remote_code=True,
        ).eval()

        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=False,
            token=os.getenv("HF_TOKEN"),
        )

        if "llama2" in model_path.lower():
            tokenizer.pad_token = tokenizer.unk_token
            tokenizer.padding_side = "left"
        if "vicuna" in model_path.lower():
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "left"
        if "qwen" in model_path.lower():
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
            tokenizer.padding_side = "left"
        if "mistral" in model_path.lower() or "mixtral" in model_path.lower():
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        if not tokenizer.pad_token:
            tokenizer.pad_token = tokenizer.eos_token

        lm = HuggingFace(model_name, model, tokenizer)

    return lm, template


def get_model_path_and_template(model_name):
    full_model_dict = {
        "gpt-4o":             {"path": "gpt-4",           "template": "gpt-4"},
        "gpt-4o-mini":        {"path": "gpt-4",           "template": "gpt-4"},
        "gpt-4-0125-preview": {"path": "gpt-4",           "template": "gpt-4"},
        "gpt-4-1106-preview": {"path": "gpt-4",           "template": "gpt-4"},
        "gpt-4":              {"path": "gpt-4",           "template": "gpt-4"},
        "gpt-3.5-turbo":      {"path": "gpt-3.5-turbo",   "template": "gpt-3.5-turbo"},
        "gpt-3.5-turbo-1106": {"path": "gpt-3.5-turbo",   "template": "gpt-3.5-turbo"},
        "vicuna":             {"path": VICUNA_PATH,        "template": "vicuna_v1.1"},
        "vicuna-13b":         {"path": VICUNA_13B_PATH,    "template": "vicuna_v1.1"},
        "llama2":             {"path": LLAMA_7B_PATH,      "template": "llama-2"},
        "llama2-7b":          {"path": LLAMA_7B_PATH,      "template": "llama-2"},
        "llama2-13b":         {"path": LLAMA_13B_PATH,     "template": "llama-2"},
        "llama2-70b":         {"path": LLAMA_70B_PATH,     "template": "llama-2"},
        "llama3-8b":          {"path": LLAMA3_8B_PATH,     "template": "llama-2"},
        "llama3.1-8b":        {"path": LLAMA31_8B_PATH,    "template": "llama-2"},
        "llama3-70b":         {"path": LLAMA3_70B_PATH,    "template": "llama-2"},
        "gemma-2b":           {"path": GEMMA_2B_PATH,      "template": "gemma"},
        "gemma-7b":           {"path": GEMMA_7B_PATH,      "template": "gemma"},
        "mistral-7b":         {"path": MISTRAL_7B_PATH,    "template": "mistral"},
        "mixtral-7b":         {"path": MIXTRAL_7B_PATH,    "template": "mistral"},
        "r2d2":               {"path": R2D2_PATH,          "template": "zephyr"},
        "phi3":               {"path": PHI3_MINI_PATH,     "template": "llama-2"},
        "qwen2.5-7b":         {"path": QWEN2_5_7B_PATH,    "template": "qwen2.5"},
        "claude-instant-1":   {"path": "claude-instant-1", "template": "claude-instant-1"},
        "claude-2":           {"path": "claude-2",         "template": "claude-2"},
        "palm-2":             {"path": "palm-2",           "template": "palm-2"},
    }
    assert model_name in full_model_dict, (
        f"Model '{model_name}' not found. Available: {list(full_model_dict.keys())}"
    )
    return full_model_dict[model_name]["path"], full_model_dict[model_name]["template"]
