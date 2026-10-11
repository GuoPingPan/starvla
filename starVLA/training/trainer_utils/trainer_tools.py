"""
trainer_tools.py
================

训练器通用工具集 (`TrainerUtils` 类 + 一组模块级 helper)。

被谁用
------
- `starVLA/training/train_starvla.py` (VLA 训练器) 继承 `TrainerUtils`,
  调用这里的 freeze / load_pretrained / setup_distributed_training 等。
- `starVLA/training/train_starvla_cotrain.py` / `train_starvlm.py` 同样继承。
- 把这些通用逻辑抽出来,是为了让每个 trainer_*.py 只关心训练循环本身。

主要内容
--------
1. CLI 参数归一化:`normalize_dotlist_args`
2. 优化器/调度器构造:`setup_optimizer_and_scheduler` + `build_param_lr_groups`
3. 进程/装饰器:`only_main_process`、`is_main_process`、`_dist_rank`
4. 图像/数据工具:`resize_images`、`euclidean_distance`
5. TrainerUtils 静态方法集合:
   - 参数冻结 / 统计:`freeze_backbones`、`print_trainable_parameters`、`print_freeze_status`
   - checkpoint 加载:`load_pretrained_backbones`、`_get_latest_checkpoint`
   - 分布式准备:`setup_distributed_training`
   - dataloader 重置:`_reset_dataloader`
   - 梯度手术 (multi-task):`compute_grad_angle_with_stats`、`pcgrad_project`
   - 在线评测:`eval_qwenpi`、`extract_json_from_string`
"""

from typing import Tuple
import re
import json
import numpy as np
import torch
import torch.distributed as dist
from transformers import get_scheduler

from accelerate.logging import get_logger

logger = get_logger(__name__)


def _dist_rank() -> int:
    """返回当前分布式 rank;未初始化时返回 0 (单进程)。"""
    return dist.get_rank() if dist.is_initialized() else 0


# === Define Tracker Interface ===
#

# utils/cli_parser.py


def normalize_dotlist_args(args):
    """把 argparse 解析出的"裸 CLI 参数"转成 OmegaConf 能吃的 dotlist。

    转换规则
    --------
    - `--x.y value`     → `x.y=value`       (带空格的 key/value)
    - `--x.y=value`     → `x.y=value`       (等号形式,原样保留)
    - `--flag`          → `flag=true`       (无值 flag,当作布尔 true)
    - 孤立的 value (不以 -- 开头且未被消费) → 丢弃

    这样最终可以直接喂给 `OmegaConf.from_dotlist` 做 yaml override。
    """
    normalized = []
    skip = False
    for i in range(len(args)):
        if skip:
            skip = False
            continue

        arg = args[i]
        if arg.startswith("--"):
            key = arg.lstrip("-")
            if "=" in key:
                # 已经是 key=value 形式,直接收
                normalized.append(key)
            elif i + 1 < len(args) and not args[i + 1].startswith("--"):
                # 下一个 token 看起来像值 (不以 -- 开头),拼成 key=value
                normalized.append(f"{key}={args[i + 1]}")
                skip = True
            else:
                # 没有后续值,当作布尔 flag
                normalized.append(f"{key}=true")
        else:
            pass  # skip orphaned values
    return normalized


def setup_optimizer_and_scheduler(model, cfg) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    """从 cfg.trainer 构造 AdamW + LR scheduler。

    支持
    ----
    - 按模块指定不同 LR:通过 `cfg.trainer.learning_rate` 下的字典,
      key 是模块路径 (如 `action_head`、`vlm.backbone`),value 是 LR。
      未列出的模块走 `base`。
    - 任意 transformers scheduler:通过 `cfg.trainer.lr_scheduler_type` 选择
      (`cosine_with_min_lr`、`constant_with_warmup`、`linear` 等)。
    """
    param_groups = build_param_lr_groups(model=model, cfg=cfg)
    # fused AdamW 需要 CUDA 可用,在 CPU/npu 环境下自动 fallback。
    # ⚠️ fused=True + DeepSpeed ZeRO checkpoint 恢复存在 state 布局错位:
    # resume 后首个"大更新"步会把 fp32 master 的部分元素写成 NaN (实测
    # patch_embedding.weight, 梯度有限且 clip 后更新量 ~1e-6, 不可能合法
    # 溢出 → 只能是 fused kernel 读到未恢复的 state 内存)。env 可关。
    fused_available = torch.cuda.is_available() and os.environ.get(
        "STARVLA_FUSED_ADAMW", "1"
    ).lower() not in ("0", "false", "no", "off")
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=cfg.trainer.learning_rate.base,
        betas=tuple(cfg.trainer.optimizer.betas),
        weight_decay=cfg.trainer.optimizer.weight_decay,
        eps=cfg.trainer.optimizer.eps,
        fused=fused_available,
    )

    if dist.is_initialized() and dist.get_rank() == 0:
        for group in optimizer.param_groups:
            logger.info(f"LR Group {group['name']}: lr={group['lr']}, num_params={len(group['params'])}")

    # transformers.get_scheduler 不接受未知 kwargs,这里直接透传 cfg 里的
    # scheduler_specific_kwargs,调用方需自行保证字段合法。
    sched_kwargs = cfg.trainer.scheduler_specific_kwargs
    lr_scheduler = get_scheduler(
        name=cfg.trainer.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=cfg.trainer.num_warmup_steps,
        num_training_steps=cfg.trainer.max_train_steps,
        scheduler_specific_kwargs=sched_kwargs,
    )

    return optimizer, lr_scheduler


def build_param_lr_groups(model, cfg):
    """(核心) 把模型参数按模块拆成多个优化器参数组,每组配独立 LR。

    典型用法:VLA 训练时希望 action_head 用较大 LR (5e-5)、VLM backbone 用
    较小 LR (1e-5)、其余走 base。直接通过 yaml 配置即可:

    .. code-block:: yaml

        trainer:
          learning_rate:
            base: 1.0e-5
            action_head: 5.0e-5
            vlm: 2.0e-5
          freeze_modules: "vlm.embeddings"

    处理顺序
    --------
    1. 解析 freeze_modules,把要冻结的参数 id 收集到 `frozen_params`。
    2. 遍历 learning_rate 里除 base 外的 key,按 dotted-path 找到对应子模块,
       把它的 (未冻结) 参数单独成组,记录到 used_params。
    3. 剩下没被任何组覆盖的参数,统一进 base 组。

    为什么用 id() 而不是 parameter 对象本身做 set?
    -----------------------------------------------
    nn.Parameter 没有实现 __hash__ 的去重语义 (默认走 identity),用 id() 显式
    表达"按对象身份去重"的意图,避免误把参数 hash 当成 value-based 去重。

    Args:
        model: nn.Module
        cfg: 需要有 cfg.trainer.learning_rate (dict) 和 cfg.trainer.freeze_modules (str)

    Returns:
        List[Dict]: torch.optim 接受的 param_groups 格式
    """

    lr_cfg = cfg.trainer.learning_rate
    base_lr = lr_cfg.get("base", 1e-4)  # default base learning rate

    freeze_modules = cfg.trainer.get("freeze_modules", "")
    if not isinstance(freeze_modules, str):
        freeze_modules = ""
    freeze_patterns = [p.strip() for p in freeze_modules.split(",") if p.strip()]

    used_params = set()
    frozen_params = set()
    param_groups = []

    # ---- 第 1 步:解析 freeze_modules,收集冻结参数 id ----
    for freeze_path in freeze_patterns:
        module = model
        try:
            for attr in freeze_path.split("."):
                module = getattr(module, attr)
            frozen_params.update(id(p) for p in module.parameters())
        except AttributeError:
            # 冻结路径不存在不致命 (可能用户配错),只 warning
            print(f"⚠️ freeze module path does not exist: {freeze_path}")
            continue

    # ---- 第 2 步:为每个非 base 的 LR 配置单独建组 ----
    for module_name, lr in lr_cfg.items():
        if module_name == "base":
            continue
        # 按 dotted-path 一层层 getattr 找到目标子模块
        module = model
        try:
            for attr in module_name.split("."):
                module = getattr(module, attr)
            # 过滤掉被冻结的参数 (它们不参与训练,放进优化器也没意义)
            params = [p for p in module.parameters() if id(p) not in frozen_params]
            if params:  # only add param group if there are trainable parameters
                param_groups.append({"params": params, "lr": lr, "name": module_name})
                used_params.update(id(p) for p in params)
        except AttributeError:
            # 注意:这里只构造了 ReferenceError 但没有 raise,属于已知 bug,
            # 不影响功能但会让错误"静默"。如需严格报错,改成 raise ReferenceError(...)。
            ReferenceError(f"⚠️ module path `{module_name}` not found in vla")

    # ---- 第 3 步:剩下未分组的参数走 base LR ----
    other_params = [p for p in model.parameters() if id(p) not in used_params and id(p) not in frozen_params]
    if other_params:
        param_groups.append({"params": other_params, "lr": base_lr, "name": "base"})

    return param_groups


import torch.distributed as dist


def only_main_process(func):
    """装饰器:被装饰的函数只在 rank 0 执行,其他 rank 直接 return None。

    用途:保存 checkpoint、写本地日志、画图等不需要每个 rank 都做的副作用。
    """

    def wrapper(*args, **kwargs):
        if dist.is_initialized() and dist.get_rank() != 0:
            return None  # non-main process does not execute
        return func(*args, **kwargs)

    return wrapper


from torchvision.ops import box_iou
from PIL import Image


def resize_images(images, target_size=(224, 224)):
    """递归地把嵌套 list 里的所有 PIL 图像 resize 到 target_size。

    保留原嵌套结构:输入是 list[list[Image]] 输出也是 list[list[Image]]。
    主要用于推理时的图像预处理,训练侧一般走 albumentations/torchvision transform。
    """
    if isinstance(images, Image.Image):  # if it is a single PIL image
        return images.resize(target_size)
    elif isinstance(images, list):  # if it is a list, recursively process each element
        return [resize_images(img, target_size) for img in images]
    else:
        raise ValueError("Unsupported image type or structure.")


import torch.distributed as dist


class TrainerUtils:
    """训练器通用工具集合 (全部为 @staticmethod,可直接调用,无需实例化)。

    被各 trainer (VLATrainer / VLAMTrainer / cotrain Trainer) 通过继承复用。
    """

    @staticmethod
    def freeze_backbones(model, freeze_modules=""):
        """(核心) 根据 dotted-path 列表冻结 model 的指定子模块。

        与 `build_param_lr_groups` 里的冻结逻辑一致:把 `freeze_modules`
        字符串按逗号切分,每段是一个相对 model 的 dotted-path (例如
        `"qwen_vl_interface"` 或 `"action_model.net"`),逐层 getattr 找到
        目标模块后,把它 (及所有子模块) 的 `requires_grad` 置为 False。

        设计选择
        --------
        - 不做正则匹配,直接走 dotted-path:配置更直观,也避免误冻结。
        - 路径不存在时只 warning 不 raise,允许配置里写"可选冻结"字段。
        - 单独成方法 (而不是合进 build_param_lr_groups):cotrain 场景下
          可能在 optimizer 构造之后才决定冻结哪些 (例如 LoRA 注入后)。

        Args:
            model: nn.Module
            freeze_modules: 逗号分隔的 dotted-path 字符串,例如
                `"vlm.backbone, action_head"`。空字符串 = 不冻结。

        Returns:
            model (同入参,就地修改)
        """
        frozen = []
        print("#"*30)
        print(freeze_modules)
        if freeze_modules and type(freeze_modules) == str:
            # split and remove whitespace
            patterns = [p.strip() for p in freeze_modules.split(",") if p.strip()] if freeze_modules else []

            for path in patterns:
                # split the "relative path" by dots, for example "action_model.net" → ["action_model", "net"]
                attrs = path.split(".")
                module = model
                try:
                    for attr in attrs:
                        module = getattr(module, attr)
                    # if the module is successfully get, freeze it and its all submodule parameters
                    for param in module.parameters():
                        param.requires_grad = False
                    frozen.append(path)
                except AttributeError:
                    # if the attribute does not exist, skip and print warning
                    print(f"⚠️ module path does not exist, cannot freeze: {path}")
                    continue

        # accelerator.wait_for_everyone()  # synchronize when distributed training
        if _dist_rank() == 0:
            print(f"🔒 Frozen modules with re pattern: {frozen}")
        return model

    @staticmethod
    def print_trainable_parameters(model):
        """打印模型总参数量与可训练参数量 (单位:百万)。

        仅 rank 0 执行,其他 rank 直接 return (无返回值)。
        用于训练开始前 sanity check:确认 freeze_modules 配置生效。
        """
        if _dist_rank() != 0:
            return
        print("📊 model parameter statistics:")
        num_params = sum(p.numel() for p in model.parameters())
        num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(
            f"# Parameters (in millions): {num_params / 10**6:.3f} Total, {num_trainable_params / 10**6:.3f} Trainable"
        )
        return num_params, num_trainable_params

    @staticmethod
    def load_pretrained_backbones(model, checkpoint_path=None, reload_modules=None):
        """(核心) 加载 checkpoint 到 model,支持全量加载和按子模块部分加载。

        两种模式
        --------
        1. 全量加载 (reload_modules=None):
           直接 `model.load_state_dict(checkpoint, strict=False)`,允许
           checkpoint 和 model 的 key 不完全匹配 (例如多了一些 buffer 或
           少了某些新加的 LoRA 参数)。strict=False 是为了支持"checkpoint
           是某个子集"或"model 比 checkpoint 多了点东西"的灵活场景。

        2. 按子模块加载 (reload_modules="path1,path2"):
           对每个 dotted-path,从 checkpoint 里取出所有以 `path.` 为前缀的
           key,剥离前缀后用 `module.load_state_dict(sub_state_dict, strict=True)`
           灌进对应子模块。strict=True 是因为这里用户明确指定了"我要重载
           这个模块",key 不匹配应该报错而不是静默忽略。

        文件格式
        --------
        根据 checkpoint_path 后缀自动判断:
        - `.safetensors` → safetensors.torch.load_file
        - 其他 → torch.load (兼容 .pt / .bin)

        Returns:
            list[str]: 成功加载的模块路径列表;全量加载时返回 `["<full_model>"]`。
            注意返回的 model 是入参的同一个对象 (就地修改),返回 list 只是记录。
        """
        if not checkpoint_path:
            return []
        if _dist_rank() == 0:
            print(f"📦 loading checkpoint: {checkpoint_path}")
        try:
            if _is_safetensors_path(checkpoint_path):
                from safetensors.torch import load_file

                checkpoint = load_file(checkpoint_path)
            else:
                checkpoint = torch.load(checkpoint_path, map_location="cpu")
        except Exception as e:
            raise RuntimeError(f"❌ loading checkpoint failed: {e}")

        loaded_modules = []

        if reload_modules:  # partial load
            module_paths = [p.strip() for p in reload_modules.split(",") if p.strip()]
            for path in module_paths:
                reload_modules = path.split(".")
                module = model
                try:
                    for module_name in reload_modules:  # find the module to modify level by level
                        module = getattr(module, module_name)
                    prefix = path + "."
                    # 把 "vlm.backbone.weight" 这种 key 剥离前缀成 "weight",
                    # 才能直接灌进 module.load_state_dict。
                    sub_state_dict = {k[len(prefix) :]: v for k, v in checkpoint.items() if k.startswith(prefix)}
                    if sub_state_dict:
                        module.load_state_dict(sub_state_dict, strict=True)
                        if _dist_rank() == 0:
                            print(f"✅ parameters loaded to module '{path}'")
                        loaded_modules.append(path)
                    else:
                        print(f"⚠️ parameters not found in checkpoint '{path}'")
                except AttributeError:
                    print(f"❌ cannot find module path: {path}")
        else:  # full load
            try:
                model.load_state_dict(checkpoint, strict=False)
                if _dist_rank() == 0:
                    print("✅ loaded <full_model> model parameters")
                loaded_modules = ["<full_model>"]
            except Exception as e:
                raise RuntimeError(f"❌ loading full model failed: {e}")
        return model

    @staticmethod
    def print_freeze_status(model):
        """逐参数打印 requires_grad 状态 (Frozen / Trainable)。

        调试用:freeze_backbones 之后跑一次,确认目标模块确实被冻结。
        比 print_trainable_parameters 更细粒度,能看到每个 parameter 的状态。
        """
        for name, param in model.named_parameters():
            status = "Frozen" if not param.requires_grad else "Trainable"
            print(f"{name:60s}  |  {status}")

    @staticmethod
    def setup_distributed_training(accelerator, *components):
        """用 accelerate.Accelerator.prepare 包装训练组件。

        传入的 components 通常是 `(model, optimizer, dataloader)`,会被
        accelerate 包装成 DeepSpeed 版本 (含 ZeRO 切片、梯度同步调度等)。
        返回顺序与输入一致。

        为什么不在 trainer.__init__ 里直接做?
        -------------------------------------
        因为 accelerate.prepare 必须在 optimizer / scheduler 构造完之后调用,
        而 cotrain 场景下可能在 prepare 之前还要插入 LoRA / EMA 等中间步骤,
        所以保留为独立的可调用方法。
        """
        prepared_components = accelerator.prepare(*components)
        return prepared_components

    @staticmethod
    def euclidean_distance(predicted: np.ndarray, ground_truth: np.ndarray) -> float:
        """L2 距离 (标量)。在线 eval 里用来衡量 predicted_action 与 GT 的差距。"""
        return np.linalg.norm(predicted - ground_truth)

    @staticmethod
    def _reset_dataloader(dataloader, epoch_counter):
        """安全地重置 dataloader 的 iterator,并推进 epoch 计数。

        关键点是 `sampler.set_epoch(epoch_counter)`:DistributedSampler 用
        epoch 编号作为 shuffle 的 seed,如果不调用,每个 epoch 各 rank 拿到
        的样本顺序完全相同,等于没 shuffle。
        """
        # 1. update epoch counter
        epoch_counter += 1

        # 2. set new epoch (distributed core)
        if hasattr(dataloader, "sampler") and callable(getattr(dataloader.sampler, "set_epoch", None)):
            dataloader.sampler.set_epoch(epoch_counter)

        # 3. create new iterator
        return iter(dataloader), epoch_counter

    @staticmethod
    def compute_grad_angle_with_stats(grads_a: list[torch.Tensor], grads_v: list[torch.Tensor]) -> Tuple[float, float]:
        """(multi-task 核心) 计算两组梯度向量之间的平均夹角 (度) 和标准差。

        用途:在 VLM + action_head 联合训练 / 多任务 cotrain 里,衡量两个
        loss 的梯度方向是否冲突。如果夹角长期 > 90°,说明两个任务在互相
        拉扯,需要 PCGrad 之类的梯度手术 (见 `pcgrad_project`) 或重新
        设计 loss 权重。

        实现细节 (注意维度截断)
        -----------------------
        高维空间下随机两个向量的夹角会快速收敛到 90° ("维度诅咒"),失去
        诊断意义。所以这里只取前 32 行 × 前 7 列做余弦,人为压到低维。
        这是工程妥协,不是数学严谨 —— 够用就好。

        Args:
            grads_a: action 路径梯度列表 (这里只用 grads_a[0])
            grads_v: vlm 路径梯度列表 (这里只用 grads_v[0])

        Returns:
            (mean_angle_deg, angle_std):平均角度 + 标准差 (度)
        """
        angle_degs = []

        # compute the cosine angle between each gradient block grads_a[0].shape = 1280, 3, 14, 14
        # grads_1 = grads_a[0][0]  # [3, 14, 14]
        # grads_2 = grads_v[0][0]
        # grads_a = grads_1.view(-1, 3)  # reshape to [196, 3]
        # grads_v = grads_2.view(-1, 3)

        # lang linear
        # reshape to 14*14, 3
        # layer
        grads_action = grads_a[0]  # [2048, 11008]
        grads_action = grads_action[
            :32, :7
        ]  # only take the first 7 elements, avoid cosim failure in high-dimensional space
        grads_vl = grads_v[0]  # [2048, 11008]
        grads_vl = grads_vl[
            :32, :7
        ]  # only take the first 32 elements, 7 dimensions, avoid cosim failure in high-dimensional space
        for g_a, g_v in zip(grads_action, grads_vl):
            dot = torch.sum(g_a * g_v)
            norm_a_sq = torch.sum(g_a * g_a)
            norm_v_sq = torch.sum(g_v * g_v)

            # avoid division by zero
            norm_a = torch.sqrt(norm_a_sq + 1e-16)
            norm_v = torch.sqrt(norm_v_sq + 1e-16)

            cos_sim = (dot / (norm_a * norm_v)).clamp(-1.0, 1.0)
            angle_rad = torch.acos(cos_sim)
            angle_deg = angle_rad * (180.0 / torch.pi)

            angle_degs.append(angle_deg.item())

        # compute the average angle and variance
        angle_degs_tensor = torch.tensor(angle_degs)
        mean_angle_deg = torch.mean(angle_degs_tensor).item()
        angle_variance = torch.sqrt(torch.var(angle_degs_tensor)).item()
        # accelerator.wait_for_everyone()
        return mean_angle_deg, angle_variance

    @staticmethod
    def pcgrad_project(grads_a: list[torch.Tensor], grads_v: list[torch.Tensor]) -> list[torch.Tensor]:
        """(multi-task 核心) PCGrad 梯度投影,抑制 grads_a 与 grads_v 之间的负迁移。

        PCGrad (Yu et al. NeurIPS 2020) 的核心思想:当两个任务的梯度点积 < 0
        (方向冲突) 时,把 grads_v 投影到 grads_a 的法平面上,消除冲突分量;
        点积 ≥ 0 时不做任何修改。

        公式
        ----
        如果 dot(g_a, g_v) < 0:
            g_v ← g_v - (dot(g_a, g_v) / ||g_a||^2) * g_a
        否则:
            g_v 保持不变

        这里把 grads_a 当作"主任务" (通常是 VLM loss),grads_v 当作"被投影的
        副任务" (通常是 action loss)。返回的是新的 grads_v 列表,grads_a 不动。

        用法:在 backward 之后、step 之前调用,把 projected grads_v 重新
        赋给对应 parameter.grad。
        """
        # first compute dot and ||grads_a||^2
        dot, norm_a_sq = 0.0, 0.0
        for g_a, g_v in zip(grads_a, grads_v):
            dot += torch.sum(g_a * g_v)
            norm_a_sq += torch.sum(g_a * g_a)

        if dot < 0:
            coeff = dot / (norm_a_sq + 1e-6)
            # projection
            grads_v = [g_v - coeff * g_a for g_a, g_v in zip(grads_a, grads_v)]

        return grads_v

    @staticmethod
    def eval_qwenpi(qwenpi, dataloader, num_batches=20):
        """对 QwenQFormerDiT 模型做在线评测,计算 IoU + action 距离。

        评测指标
        --------
        - IoU:模型输出的 pick/place bbox 与 GT 的 IoU。需要模型支持
          CoT 输出 (predict_action_withCoT),返回 JSON 字符串。
        - action 距离:归一化后的 predicted action 与 GT action 的欧氏距离,
          除以元素数得到 per-element 平均距离。

        Args:
            qwenpi: QwenQFormerDiT 模型实例
            dataloader: 评测用的 dataloader (每个 batch 是 example dict 列表)
            num_batches: 评测 batch 数,默认 20

        Returns:
            {"iou_scores": [...], "average_action_distance": float}
        """
        iou_scores = []
        action_distances = []
        count = 0

        dataset_iter = iter(dataloader)
        while count < num_batches:
            try:
                batch_samples = next(dataset_iter)
                count += 1
            except StopIteration:
                break

            # extract data
            images = [example["image"] for example in batch_samples]
            instructions = [example["lang"] for example in batch_samples]
            actions = [example["action"] for example in batch_samples]
            solutions = [example["solution"] for example in batch_samples]

            # model prediction
            predicted_solutions, normalized_actions = qwenpi.predict_action_withCoT(
                images=images, instructions=instructions, use_ddim=False, num_ddim_steps=20
            )

            # extract and convert predicted results
            parsed_solutions = []
            for solution in predicted_solutions:
                parsed_solution = TrainerUtils.extract_json_from_string(solution)
                parsed_solutions.append(parsed_solution)

            # compute IoU
            for pred_dict, gt_dict in zip(parsed_solutions, solutions):
                pred_pick_bbox = torch.tensor(pred_dict["pick"]["bbox_2d"], dtype=torch.float32).unsqueeze(0)
                gt_pick_bbox = torch.tensor(gt_dict["pick"]["bbox_2d"], dtype=torch.float32).unsqueeze(0)
                pred_place_bbox = torch.tensor(pred_dict["place"]["bbox_2d"], dtype=torch.float32).unsqueeze(0)
                gt_place_bbox = torch.tensor(gt_dict["place"]["bbox_2d"], dtype=torch.float32).unsqueeze(0)

                pick_iou = box_iou(pred_pick_bbox, gt_pick_bbox).item()
                place_iou = box_iou(pred_place_bbox, gt_place_bbox).item()

                iou_scores.append({"pick_iou": pick_iou, "place_iou": place_iou})

            # compute action distance
            actions = np.array(actions)  # convert to numpy array
            num_pots = np.prod(actions.shape)  # B*len*dim
            action_distance = TrainerUtils.euclidean_distance(normalized_actions, actions)
            average_action_distance = action_distance / num_pots
            action_distances.append(average_action_distance)

        # summarize results
        avg_action_distance = np.mean(action_distances)
        return {"iou_scores": iou_scores, "average_action_distance": avg_action_distance}

    @staticmethod
    def extract_json_from_string(input_string):
        """从可能含噪声的字符串里提取第一个 JSON 对象。

        用于解析 LLM CoT 输出:模型经常在 JSON 前后输出 markdown 标记、
        自然语言解释等,需要从中抠出 `{...}` 部分。

        匹配规则:`re.search(r"{.*}", ..., re.DOTALL)`,贪婪匹配第一个
        `{` 到最后一个 `}`。如果模型输出了多个 JSON 对象,会取整体范围
        (可能导致 JSONDecodeError)。
        """
        json_match = re.search(r"{.*}", input_string, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                print(f"JSON decode failed: {e}")
                return None
        else:
            print("No valid JSON part found")
            return None

    def _get_latest_checkpoint(self, checkpoint_dir):
        """在 checkpoint_dir 里找 step 数最大的 checkpoint 文件。

        命名约定 (与 _save_checkpoint 配套):
            steps_{N}_pytorch_model.pt
            steps_{N}_model.safetensors

        其中 N 是训练步数。函数会正则匹配所有符合命名的文件,提取 N,
        排序后返回步数最大的那个。

        Returns:
            (latest_checkpoint_path, completed_steps)
            如果目录不存在或没找到任何 checkpoint,返回 (None, 0)。

        注意:本方法依赖模块底部的 `import os` (L541),属于已知代码顺序
        问题 —— 实际运行没问题 (Python 在调用方法时 os 已 import 完毕),
        但静态检查会报警告。重构时建议把 import 提到文件顶部。
        """
        if not os.path.exists(checkpoint_dir):
            self.accelerator.print(f"No checkpoint directory found at {checkpoint_dir}")
            return None, 0

        # Find all checkpoints matching the naming convention, supports .pt and .safetensors
        checkpoints = [
            f for f in os.listdir(checkpoint_dir)
            if re.match(r"steps_(\d+)_(?:pytorch_model\.pt|model\.safetensors)$", f)
            and os.path.isfile(os.path.join(checkpoint_dir, f))  # ensure it is a file
        ]

        if not checkpoints:
            self.accelerator.print(f"No checkpoints found in {checkpoint_dir}")
            return None, 0

        # Extract step numbers and sort
        try:
            checkpoints_with_steps = [
                (ckpt, int(re.search(r"steps_(\d+)_(?:pytorch_model\.pt|model\.safetensors)$", ckpt).group(1)))
                for ckpt in checkpoints
            ]
        except AttributeError as e:
            self.accelerator.print(f"Error parsing checkpoint filenames: {e}")
            return None, 0

        # Sort by step number and get the latest checkpoint
        checkpoints_with_steps.sort(key=lambda x: x[1])
        latest_checkpoint, completed_steps = checkpoints_with_steps[-1]

        latest_checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
        self.accelerator.print(f"Latest checkpoint found: {latest_checkpoint_path}")
        return latest_checkpoint_path, completed_steps

import os


def is_main_process():
    """通过环境变量 RANK 判断是否主进程。

    与 `_dist_rank` 区别:这个不依赖 torch.distributed 已初始化,纯粹
    读环境变量。在 accelerate 把 dist 初始化好之前,如果想判断主进程
    (例如 setup_directories 阶段),用这个更稳。
    """
    rank = int(os.environ.get("RANK", 0))  # if RANK is not set, default to 0
    return rank == 0


def _is_safetensors_path(path):
    """根据后缀判断是否 safetensors 文件 (.safetensors)。"""
    return str(path).endswith(".safetensors")
