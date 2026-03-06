import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from src.model import LoRALayer  
from src.config import Config

def build_model():
    # 这个函数负责“搭积木”：
    # 1) 按配置加载基础大模型
    # 2) 设置精度与量化
    # 3) 准备 tokenizer
    # 4) 在必要时把模型移动到对应硬件
    print(f"Loading Base Model: {Config.BASE_MODEL_ID}...")

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32
    }
    dtype_str = getattr(Config, "TORCH_DTYPE", "float16")
    target_dtype = dtype_map.get(dtype_str, torch.float16)

    # 根据配置决定是否启用 4bit 量化。
    # 量化的好处：显存占用更低，加载更快；
    # 但通常需要 NVIDIA CUDA 环境（bitsandbytes 对 AMD/CPU 支持有限）。
    if Config.USE_4BIT and Config.DEVICE == "cuda":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=target_dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    else:
        quantization_config = None

    # 设置 device_map：
    # - 非 dml：使用 "auto"，让 Transformers 自动把权重放到合适设备（大模型常用）
    # - dml：先不自动分配，后面手动迁移到 DirectML 设备
    if Config.DEVICE == "dml":
        device_map = None
    else:
        device_map = "auto"

    model = AutoModelForCausalLM.from_pretrained(
        Config.BASE_MODEL_ID,
        device_map=device_map,
        torch_dtype=target_dtype,
        quantization_config=quantization_config
    )

    tokenizer = AutoTokenizer.from_pretrained(Config.BASE_MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # DirectML 特殊处理：模型加载后再手动迁移。
    if Config.DEVICE == "dml":
        import torch_directml
        device = torch_directml.device()
        print(f"Moving model to DirectML (AMD GPU)")
        model.to(device)
    
    return model, tokenizer

# 在基础模型中注入 LoRA 层。
# 直观理解：把原始线性层“包裹”成「原层 + 小适配器」，从而只训练小参数。
def inject_lora_layers(model):
    count = 0

    # 读取需要替换的模块名后缀。
    # 例如 q_proj / k_proj / v_proj 等注意力与前馈层投影。
    target_modules = set(Config.LORA_TARGET_MODULES)

    # 遍历模型里的所有子模块（模型结构可理解为一棵树）。
    for name, module in model.named_modules():

        # 用“模块名最后一段”判断是否命中目标。
        # 例如 "model.layers.0.self_attn.q_proj" 的后缀是 "q_proj"。
        module_suffix = name.split('.')[-1]

        if module_suffix in target_modules:

            # parent_name: 父模块路径；child_name: 需要替换的子模块名
            parent_name = name.rsplit('.', 1)[0]
            child_name = name.rsplit('.', 1)[1]

            parent_module = model.get_submodule(parent_name)

            # 构造 LoRA 包装层：内部保留原层，并新增可训练的低秩参数。
            lora_layer = LoRALayer(
                original_layer=module, 
                rank=Config.LORA_R,
                alpha=Config.LORA_ALPHA,
                dropout=Config.LORA_DROPOUT
            )

            # 保证新层与原层在同一设备上（避免 device mismatch 报错）。
            lora_layer.to(module.weight.device)

            # 关键替换：把父模块里的原子层，替换成 lora_layer。
            setattr(parent_module, child_name, lora_layer)
            count += 1
            
    print(f"LoRA injected modules: {count}")
    return model
