
import torch
import torch.nn as nn
import math
import os
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import get_cosine_schedule_with_warmup
from datasets import load_dataset
from tqdm import tqdm
from src.model import LoRALayer
from src.loader import build_model, inject_lora_layers 
from src.config import Config 

'''
训练脚本说明（新手向）：
1) 先加载基础模型，并注入 LoRA 小模块；
2) 冻结原模型参数，只训练 LoRA 参数；
3) 准备数据，并且把“提示词部分”label 屏蔽（设为 -100，不计入损失）；
4) 进行梯度累积训练，周期性保存 checkpoint；
5) 最后仅保存 LoRA 权重，体积更小，部署更方便。
'''

def freeze_parameters(model):
    '''
    冻结参数函数：
    - 名称中包含 lora_a / lora_b 的参数：保留可训练；
    - 其余参数：全部冻结。

    这样训练会非常省资源，因为只更新很小一部分参数。
    '''
    print("Freezing Base Model Parameters...")
    frozen_count = 0
    lora_count = 0
    
    for name, param in model.named_parameters():
        if "lora_a" in name or "lora_b" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
   
    return model


def print_trainable_parameters(model):
    ''' 
    打印可训练参数占比，帮助你确认：
    “我们是不是只在训练 LoRA 小参数”。
    '''
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())

    print(
        f"Trainable Parameters: {trainable_params} || " 
        f"All Parameters: {all_params} || "
        f"Percentage: {100 * trainable_params / all_params:.2f}%"
    )

def format_and_mask(sample, tokenizer):
    '''
    把单条样本转成模型输入，并做标签掩码（mask）：
    - 输入文本 = 提示词 + 正确密码；
    - labels 默认复制 input_ids；
    - 但提示词部分的 labels 会改成 -100，表示“这里不参与损失计算”；
    - padding 位置也设成 -100。

    结果：模型主要学习“如何生成密码”，而不是重复提示词模板。
    '''
    full_text = Config.get_formatted_input(
        pii_dict=sample['pii'], 
        target_password=sample['output'],
        tokenizer=tokenizer
    )

    max_len = getattr(Config, 'MAX_SEQ_LENGTH', 512)
    encodings = tokenizer(full_text, truncation=True, padding='max_length', max_length=max_len)

    input_ids = encodings['input_ids']
    attention_mask = encodings['attention_mask']
    labels = list(input_ids)

    prompt_text = Config.get_formatted_input(
        pii_dict=sample['pii'], 
        target_password=None,
        tokenizer=tokenizer
    )
    
    prompt_ids = tokenizer(prompt_text, truncation=True, max_length=max_len)["input_ids"]
    prompt_len = len(prompt_ids)

    if prompt_len < len(labels):
        labels[:prompt_len] = [-100] * prompt_len

    for i, mask_val in enumerate(attention_mask):
        if mask_val == 0:
            labels[i] = -100

    return {
        "input_ids": input_ids,
        "attention_mask": encodings["attention_mask"],
        "labels": labels
    }

def prepare_data(tokenizer):
    # 读取 json 数据集，并对每条样本调用 format_and_mask。
    # 最后返回 DataLoader，供训练循环按批次读取。
    print("Processing Data with Masking...")

    dataset = load_dataset("json", data_files=str(Config.RAW_DATA_FILE), split="train")

    tokenized_dataset = dataset.map(
        lambda x: format_and_mask(x, tokenizer),
        remove_columns=dataset.column_names  
    )
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    return DataLoader(tokenized_dataset, Config.TRAIN_BATCH_SIZE, shuffle=True)

def train_loop(model, tokenizer, dataloader):
    '''
    训练主循环：
    - 优化器：AdamW；
    - 学习率调度：cosine + warmup；
    - 梯度累积：用更小显存模拟更大 batch；
    - 每个 epoch 结束保存 checkpoint。
    '''
    print("Starting Training Loop...")
    global_step = 0
    num_training_steps = (len(dataloader) // Config.GRAD_ACCUMULATION) * Config.NUM_EPOCHS
    
    checkpoint_every_steps = getattr(Config, 'CHECKPOINT_EVERY_STEPS', 0)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr = Config.LEARNING_RATE
    )

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.2 * num_training_steps), 
        num_training_steps=num_training_steps
    )

    model.train()  

    for epoch in range(Config.NUM_EPOCHS): 
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        
        optimizer.zero_grad()

        for step, batch in enumerate(progress_bar):

            # 把当前 batch 放到模型所在设备（如 GPU）。
            input_ids = batch['input_ids'].to(model.device)
            mask = batch['attention_mask'].to(model.device)
            labels = batch['labels'].to(model.device)

            outputs = model(input_ids=input_ids, attention_mask=mask, labels=labels)

            # 梯度累积核心：把 loss 除以累积步数，避免梯度量级变大。
            loss = outputs.loss / Config.GRAD_ACCUMULATION
            loss.backward()

            if (step + 1) % Config.GRAD_ACCUMULATION == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                
                # 如果配置了更高频 checkpoint，则在这里额外保存。
                if checkpoint_every_steps > 0 and global_step % checkpoint_every_steps == 0:
                    save_checkpoint(model, optimizer, scheduler, epoch, global_step, current_loss)

            current_loss = loss.item() * Config.GRAD_ACCUMULATION
            total_loss += current_loss
            progress_bar.set_postfix(loss=current_loss)
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} Complete. Average Loss: {avg_loss}")
        
        save_checkpoint(model, optimizer, scheduler, epoch, global_step, avg_loss)

    return model

def save_checkpoint(model, optimizer, scheduler, epoch, global_step, loss):
    # checkpoint 保存训练“中间状态”，支持断点续训。
    checkpoint_dir = Config.MODELS_DIR / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    
    checkpoint_path = checkpoint_dir / f"checkpoint_epoch{epoch+1}_step{global_step}.pt"
    
    # 只保存 LoRA 参数，减小文件体积。
    lora_state_dict = {k: v for k, v in model.state_dict().items() if "lora_" in k}
    
    checkpoint = {
        'epoch': epoch,
        'global_step': global_step,
        'lora_state_dict': lora_state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
    }
    
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path.name}")
    
    # 仅保留最近 3 个 checkpoint，避免磁盘被占满。
    checkpoints = sorted(checkpoint_dir.glob("checkpoint_*.pt"), key=lambda x: x.stat().st_mtime)
    for old_ckpt in checkpoints[:-3]:
        old_ckpt.unlink()
        print(f"Removed old checkpoint: {old_ckpt.name}")

def save_model(model):
    '''
    保存最终训练结果：
    - 不保存基础大模型（太大）；
    - 只保存 LoRA 参数（轻量）。
    '''
    print(f"Saving LoRA Weights to {Config.WEIGHTS_FILE}...")
    lora_state_dict = {k: v for k, v in model.state_dict().items() if "lora_" in k}
    torch.save(lora_state_dict, Config.WEIGHTS_FILE)
    print("Success!")

if __name__ == "__main__":
    # 1) 构建基础模型与 tokenizer
    model, tokenizer = build_model()
    # 2) 注入 LoRA 层
    model = inject_lora_layers(model)
    # 3) 冻结非 LoRA 参数
    model = freeze_parameters(model)
    # 4) 可选：开启 gradient checkpointing，进一步省显存
    if getattr(Config, 'USE_GRADIENT_CHECKPOINTING', False):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        print("Gradient checkpointing enabled (saves VRAM)")
    # 5) 打印可训练参数占比，做 sanity check
    print_trainable_parameters(model)
    # 6) 准备数据
    dataloader = prepare_data(tokenizer)
    # 7) 开始训练
    model = train_loop(model, tokenizer, dataloader)
    # 8) 保存 LoRA 权重
    save_model(model)
