# FastWAM-CommonBase

为四个 LIBERO 专家建立**同一个完整参数起点**的独立训练仓库。
包含实际 Fast-WAM 模型、数据处理、训练器、共同 base 严格加载与审计代码；不含权重、数据集、旧实验队列或 TCR 融合实现。
目录扁平：模型和训练器位于 `fastwam/`，不再套 `src/src`。

**状态：代码与 CPU 回归测试已验证；尚未在全新环境完成大权重下载及真实多卡训练。不要直接排四轮长训练，先做下面的验收。**
共同 checkpoint 仍待选择；没有开始训练或删除旧专家。GitHub 仅发布代码与文档，不含权重和数据。

## 1. 先分清三种权重

| 名称 | 内容 | 能否直接作为四专家完整共同起点 |
|---|---|---|
| Wan2.2-TI2V-5B | 视频预训练主干、VAE、T5 | 不能：没有完整机器人动作/状态接口 |
| ActionDiT 插值文件 | 从视频主干转换的动作主干，不含新建接口 | 不能 |
| sealed common base | 完整 `mot`（含动作接口）及 `proprio_encoder`，同时锁定外部冻结资源 | 可以；四专家必须加载同一份 |

你有两个明确选择，**只能选一个**：

- 从 Wan 起步：只初始化一次完整 Fast-WAM，保存为共同 base，四个分支从它重新训练。
- 保留已有训练成果：明确选一份已有完整 Fast-WAM checkpoint，严格校验后作为四个新分支共同起点。这是 weights-only warm start，步数从 0 开始，不继承它的优化器。

第二种不会丢掉旧文件，但只继承被选中那一份的学习成果。让原来的四份专家各自续训，不能追溯修复历史上不同的完整初始化。
官方发布的 LIBERO policy 已做过任务训练，不能称为“没有接触 LIBERO 的预训练 base”；若选它，实验需记录这段数据暴露。

## 2. 环境

推荐 Linux、Python 3.10、支持 BF16 的 NVIDIA GPU。此包配置为 PyTorch 2.7.1 + CUDA 12.8、
torchvision 0.22.1，详细直接依赖见 `pyproject.toml`；传递依赖并非完整 lockfile。
需要兼容驱动、FFmpeg、C/C++ 编译工具；DeepSpeed 的运行时编译通常还需要与 Torch 匹配的 CUDA toolkit 和 `CUDA_HOME`。
CUDA wheel 不能代替显卡驱动或编译工具链。

所有命令在仓库根目录运行；路径避免空格、逗号和 Hydra 特殊字符。安装脚本默认仅预览：

```bash
bash scripts/install.sh
bash scripts/install.sh --execute
source .venv/bin/activate
python -m pip check
python scripts/doctor.py
python -m unittest discover -s tests -v
```

安装不会覆盖已有 `.venv`。不要把旧机器的 `.python-packages` 或全局 `PYTHONPATH` 混进新环境。
不提供“某张卡肯定装得下”的保证：ZeRO-1 只分片优化器状态，每卡仍保留模型/梯度与激活；
先用小 batch 实测峰值。CPU 转换/封存也会加载大模型，需要充足主存。

## 3. 下载：模型、tokenizer、LIBERO 数据

官方来源：

- [Wan2.2-TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B)
- [Wan2.1 tokenizer](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B)
- [Fast-WAM 的 LIBERO 数据](https://huggingface.co/datasets/yuanty/LIBERO-fastwam)
- 可选 [已训练 LIBERO policy](https://huggingface.co/yuanty/fastwam)

下载脚本使用 `configs/assets.lock.json` 中固定的 commit，不跟随可变的 main。
Wan 模型资源约 34 GB，此外还有数据、ActionDiT、完整 base、每个专家及含优化器状态的训练快照；
**代码包很小不意味着训练磁盘开销小**，请先查看磁盘容量，训练快照可能成为最大占用。

```bash
python scripts/download_assets.py wan tokenizer libero
python scripts/download_assets.py wan tokenizer libero --execute
python scripts/extract_libero.py
python scripts/extract_libero.py --execute
```

数据只使用仓库提供的四个旧格式 `libero_*_no_noops_lerobot.tar.gz`，**不混用 `lerobot_v30/`**。
Long 对应 `libero_10_no_noops_lerobot`。解压器拒绝路径穿越、链接及已有目标目录。
下载/解压原始数据集不会连接真机，也不是在线成功率评测。

只有确定要用官方**已训练** policy 时，另运行
`python scripts/download_assets.py released-policy --execute`；
文件下载到 `checkpoints/released-policy/`，不会自动被选为共同 base。

本包使用官方 `Wan2.2_VAE.pth`，不依赖另一仓库的转换版 VAE。
该下载路径已核对，源代码具有原始 VAE 转换逻辑；本次未下载该大文件做完整加载测试。
准备和训练阶段关闭自动下载，缺文件应修复下载，而不是偷偷换成其他版本。

## 4. 准备动作主干和文本缓存

```bash
python scripts/workflow.py prepare action
python scripts/workflow.py prepare action --execute
python scripts/workflow.py prepare text --gpus 4
python scripts/workflow.py prepare text --gpus 4 --execute
python scripts/doctor.py --assets --data
```

ActionDiT 转换在 CPU 执行，不是训练；输出已有时拒绝覆盖。
文本预计算会占用指定 GPU，按四个套件的任务描述缓存，已有缓存不覆盖。
缓存损坏或模型版本不一致时不要沿用；可另行备份原缓存后重新准备。
`doctor` 仅检查导入/文件，不证明缓存逐条完整或模型能成功训练。

## 5. 建立一份完整共同 base

下面两个方案只执行其中一个。未加 `--execute` 时仅预览。

### A：从 Wan 构造新完整起点

```bash
python scripts/workflow.py seal --fresh-wan --manifest artifacts/common-base/base.json
python scripts/workflow.py seal --fresh-wan --manifest artifacts/common-base/base.json --execute
```

固定初始化 seed=3407，**在创建模型前设定**。动作和状态接口只在这里新建一次，
完整权重写到 `artifacts/common-base/common-base.pt`，并严格加载回读验证。

### B：用一份已有完整 checkpoint

将下面路径换成你最终选定的、可信的完整权重；当前没有预选：

```bash
python scripts/workflow.py seal --checkpoint /ABS/PATH/CHOSEN_FULL_POLICY.pt --manifest artifacts/common-base/base.json
# 确定选择后，在同一命令末尾加 --execute
```

不复制或改写源权重。此 manifest 会引用该文件，迁移机器时必须同时传它，或在新机器重新封存同一文件。
不要用 pickle 权重加载不可信来源的文件。

封存会记录 checkpoint SHA256、完整模型参数/缓冲区指纹、外部模型资源哈希和配置哈希。
训练前再次校验；模型加载后还会比对封存时的完整指纹，任何缺项、错形状、非有限值或漂移都停止。
因此 hash/读回检查可能很慢，**不能为了省启动时间绕过它**。
修改模型或数据配置后不能继续沿用旧 manifest 混训。

## 6. 必须先做小规模验收

先选空闲 GPU，下面 `4,5,6,7` 只是示例，不代表当前空闲。
每 GPU batch=1，4 卡全局 batch=4，开启梯度检查点。正式配置须按测得显存决定。

```bash
python scripts/workflow.py train --suite spatial --manifest artifacts/common-base/base.json --output runs/smoke-spatial --gpus 4,5,6,7 --steps 2 --batch-size 1 --save-every 2
# 确认打印的命令后，在末尾加 --execute 才实际训练
```

冒烟验收不是只看到进程起来：

1. 每个 rank 均生成 `common-base-init-rankN.json`，没有随机补参警告/哈希不一致。
2. 完成两步，loss 有限，未 OOM；保存 `checkpoints/weights/step_000002.pt`。
3. 同时存在 `checkpoints/state/step_000002/` 和其 `trainer_state.json`。
4. 再用一个总步数更大的独立短 run，人工中断到已保存的中间步，然后按下一节恢复；
   确认从对应 global_step 继续，优化器/调度器和数据游标确实恢复。

已经到达 max_steps 的 2-step run 不会凭空继续；不要通过改步数“冒充”断点恢复测试。
训练失败保留输出用于排查，重试请用新的 output 目录，不能覆盖旧结果。

## 7. 四专家正式训练

四次调用**完全相同的 manifest**，只改 suite 和独立 output。
下面 12,000 步是命令示例，不是已经确认的最优训练预算，也不保证复现历史最佳成功率。
可先运行不带 `--execute` 的版本查看计划；以下示例明确会启动训练：

```bash
for suite in spatial object goal long; do
  python scripts/workflow.py train \
    --suite "$suite" --manifest artifacts/common-base/base.json \
    --output "runs/commonbase-$suite" --gpus 4,5,6,7 \
    --steps 12000 --batch-size 1 --grad-accum 1 --save-every 1000 \
    --execute || break
done
```

这是**串行**四任务，失败即停止，不会偷偷启动四组竞争相同 GPU 的任务。
默认 lr=1e-4、weight_decay=1e-2、BF16、cosine schedule；训练 seed 默认3407。
batch 参数是每卡值，全局 batch = batch × 卡数 × grad-accum。
当前小 batch 示例改变了历史训练的全局 batch，不应称为原超参数复现；需要复现时先核对旧配置再设置。
不自动删除中间快照，不包含 rollout 成功率评测或“自动挑选最好 checkpoint”。

四分支均启动并产生完整初始化回执后检查：

```bash
python scripts/check_common_base_receipts.py \
  runs/commonbase-spatial runs/commonbase-object runs/commonbase-goal runs/commonbase-long
```

必须输出 `accepted: true`；它证明完整初始化一致，**不证明训练成功率或融合有效**。
每个套件保留各自的数据归一化统计 `dataset_stats.json`；共同参数起点不等于归一化完全一致。
后续融合/评测仍须显式匹配 stats，不能只交四个权重就假定动作尺度一致。

## 8. 同一分支的断点恢复

这里只接受本入口创建的 run 自己的**完整 state 目录**，拒绝另一专家的 state 或纯 weights 文件。
保持原卡数、batch、步数、优化器配置；GPU 物理编号可以换。下面路径需换成实际存在的快照：

```bash
python scripts/workflow.py resume --output runs/commonbase-spatial \
  --state runs/commonbase-spatial/checkpoints/state/step_001000 --gpus 4,5,6,7
# 核对后加 --execute
```

恢复命令使用已保存的训练配置，清空 init_checkpoint，恢复优化器/调度器/步数/数据游标。
不要同时使用 init_checkpoint 和 resume。不要在同一 output 同时运行两个进程。
底层训练器还有兼容旧任务的宽松加载入口；本项目的共同 base 流程**不要绕过 workflow.py 去用它**。

## 9. 交付与边界

迁移代码：整个仓库或旁边的代码压缩包即可；安装后下载资源。
压缩包不带 Git 历史，解压后可用 `git init -b main` 新建本地仓库；
不要只传测试构建出的 wheel，它不包含完整训练脚本和配置目录。
迁移共同 base：manifest、被引用的完整 checkpoint、相同配置、匹配哈希的 assets/ActionDiT。
送服务器后续融合：四专家所选 `.pt`、每个 run 的 `config.yaml`、`dataset_stats.json`、
`launch.json`、全部初始化回执和共同 base manifest；最好同时保留共同 base 本体。
需要续训时再传完整 `checkpoints/state/step_*/`，仅 weights 不足以恢复优化器。

本仓库在论文编译目录旁，**清理 paper-build 时须保留或先移出本仓库**。
详细必传/选传清单、目录结构及打包命令见 [上传与跨机器交接清单](docs/TRANSFER_GUIDE.md)。
另见 [验证记录](docs/VALIDATION.md)、[来源与修改范围](docs/PROVENANCE.md)。
代码基于 [FastWAM](https://github.com/yuantianyuan01/FastWAM)，保留其 MIT 许可及第三方原始许可头。
