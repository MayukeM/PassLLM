import torch
from src.generation_engine import dynamic_beam_search
from src.config import Config  
import copy
import random
from tqdm import tqdm


def predict_password(model, tokenizer, target_profile, max_depth=16, beam_schedule=None):
    """
    密码预测主函数（纯逻辑，不涉及 Web 界面）。

    Args:
        model: 已加载好的语言模型
        tokenizer: 对应分词器
        target_profile: 用户信息字典（PII）
    Returns:
        候选密码列表（每项通常含 password / score）
    """

    if max_depth is None:
        max_depth = Config.MAX_PASSWORD_LENGTH
        
    if beam_schedule is None:
        beam_schedule = Config.SCHEDULE_STANDARD

    # 存放所有轮次生成出的候选结果（后续会去重和排序）
    all_candidates = []
    keys = list(target_profile.keys())
    num_runs = getattr(Config, 'INFERENCE_NUM_RUNS', 5)
    target_keep_ratio = getattr(Config, 'INFERENCE_KEEP_RATIO', 0.8)
    start_ratio = (target_keep_ratio * 0.2) + 0.8

    print(f"\n[+] Running {num_runs} inference passes with {target_keep_ratio*100:.0f}% of fields randomly masked each time...")

    pbar = tqdm(range(num_runs), desc="Inference runs", ncols=80)
    pbar.update(0)  # Show bar immediately

    for run_idx in pbar:
        # progress 从 0 -> 1，表示当前是第几轮推理。
        # 后面会用它来动态调整一些策略（如 score_penalty）。
        progress = run_idx / max(1, num_runs - 1)
        current_keep_ratio = start_ratio + (target_keep_ratio - start_ratio) * progress

        # 深拷贝，避免修改原始输入数据。
        new_profile = copy.deepcopy(target_profile)

        # 字段变体替换：按概率用“别名字段”覆盖“主字段”。
        # 例如用户可能把 work_email 当成 email 使用。
        for main_field, variant_field, prob in Config.field_variations:
            if random.random() < prob:
                if variant_field in new_profile:
                    new_profile[main_field] = new_profile[variant_field]

        # 清理未知字段，只保留 schema 里定义过的键。
        for existing_field in list(new_profile.keys()):
            if existing_field not in Config.schema_defaults:
                del new_profile[existing_field]

        total_keys = len(new_profile)
        profile_partial = {}

        # 随机保留部分字段，模拟现实中信息缺失/不完整的情况。
        for i, (key, value) in enumerate(new_profile.items()):

            if random.random() < current_keep_ratio:
                if isinstance(value, list):
                    # 列表字段（如 sister_pw）按元素级别再次随机保留。
                    filtered_items = []
                    for item in value:
                        if random.random() < current_keep_ratio:
                            filtered_items.append(item)
                    profile_partial[key] = filtered_items
                else:
                    profile_partial[key] = value


        # 把“部分信息”拼成提示词，送入动态束搜索。
        prompt_text = Config.get_formatted_input(profile_partial, tokenizer=tokenizer)
        input_ids = tokenizer(prompt_text, return_tensors="pt")["input_ids"].to(model.device)
        candidates = dynamic_beam_search(
            model=model,
            tokenizer=tokenizer,
            auxiliary_info_ids=input_ids,
            max_depth=max_depth,
            batch_size=Config.INFERENCE_BATCH_SIZE,
            beam_width_schedule=beam_schedule,
            score_penalty=progress * 0.5
        )
        all_candidates.extend(candidates)

    # 去重逻辑：同一密码只保留分数更高的那条。
    seen = set()
    deduped_candidates = []
    for cand in all_candidates:
        pw = cand['password'] 
        if isinstance(pw, str) and pw.strip() and pw not in seen:
            deduped_candidates.append(cand)
            seen.add(pw)
        elif isinstance(pw, str) and pw.strip() and pw in seen:
            existing_cand = next((c for c in deduped_candidates if c['password'] == pw), None)
            if existing_cand and cand.get('score', 0) > existing_cand.get('score', 0):
                deduped_candidates.remove(existing_cand)
                deduped_candidates.append(cand)
                seen.add(pw)

    # 按分数从高到低排序，越靠前越优先。
    deduped_candidates.sort(key=lambda x: x.get('score', 0), reverse=True)
    return deduped_candidates
