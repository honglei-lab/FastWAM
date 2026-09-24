# 上传与跨机器交接清单

先区分目的地：**GitHub 只放代码；模型/数据传训练机器或融合服务器，不提交 Git。**
当前代码仓库：https://github.com/honglei-lab/FastWAM

## 1. 上传 GitHub：本次发布内容

必传：

- `fastwam/`：模型、数据处理、训练器、共同初始化检查。
- `configs/`：包含 `configs/data/`、模型/任务配置、DeepSpeed 配置和固定下载版本。
- `scripts/`、`tests/`。
- `README.md`、`docs/`、`pyproject.toml`、`LICENSE`、`.gitignore`、`SOURCE_SHA256.json`。

不传 GitHub：

- `assets/`、根目录 `data/`、`checkpoints/`、`runs/`、`artifacts/`。
- `*.pt`、`*.pth`、`*.safetensors`、训练 state、视频、数据压缩包。
- `.venv/`、缓存、编译产物、论文 PDF、旧实验代码/日志。
- token、SSH 私钥、`.env` 或云服务凭据。

注意：`configs/data/` 是**小型配置，必须提交**；根目录 `data/` 才是忽略的大数据。
没有配置大文件存储，不要通过 `git add -f` 把权重塞进仓库。

新机器获得代码：

```bash
git clone https://github.com/honglei-lab/FastWAM.git
cd FastWAM
python scripts/source_manifest.py --check
```

## 2. 换机器开始训练：有网与离线两种情况

### 新机器能下载

只需要上述代码。按 README 安装环境，运行固定版本下载、解压、ActionDiT 和文本缓存准备。
然后明确选择 fresh Wan 或某一份已有完整 policy，建立一次共同 base。

如果要继承旧训练，另传**最终选定的那一份完整 `.pt`**；不要把四个旧专家各自当成新分支的共同 base。
当前尚未替你选这份文件。

### 新机器不能下载，或要复制已封存的同一起点

| 文件/目录 | 是否必需 | 作用 |
|---|---|---|
| 完整代码仓库 | 必须 | 模型、处理逻辑和训练配置保持一致 |
| `assets/Wan-AI/Wan2.2-TI2V-5B/` | 必须 | 固定 Wan 主干、原始 VAE、T5 等 |
| `assets/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl/` | 必须 | 对应 tokenizer |
| `checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt` | 必须 | 固定的动作主干初始化资源 |
| `data/libero/` 下四个套件 | 必须 | 已解压数据，包含 meta、parquet、视频 |
| `data/text_embeds_cache/libero/` | 传已有缓存，或到新机重算 | 四个套件的文本 embedding |
| `artifacts/common-base/base.json` | 复用已封存 base 时必须 | 文件哈希、模型指纹、配置身份 |
| base.json 实际引用的完整 checkpoint | 复用已封存 base 时必须 | 仅传 JSON 不包含任何模型权重 |
| `data/downloads/*.tar.gz` | 已传解压数据时不必传 | 原始数据备份 |

fresh Wan 方案的 checkpoint 通常是 `artifacts/common-base/common-base.pt`，连同 JSON 一起传即可。
已有 policy 方案的 JSON 引用可能指向仓库外部，**必须查看其中的 checkpoint 字段并带上实际文件**。
最稳妥的方法是在新机器对同一份完整权重重新运行 seal，核对原/新
`checkpoint_sha256` 和 `initial_model_sha256` 完全一致，再启动四个分支。
不要直接改 JSON 中的哈希来绕过校验。

当前入口会校验封存清单里的所有模型资源，即使某资源在缓存训练时不会实际加载，也不能随意删掉。
环境建议按脚本重建，不直接复制旧 `.venv`。

## 3. 四专家训练完成：传融合服务器的文件

先通过初始化审计，确认输出 `accepted: true`：

```bash
python scripts/check_common_base_receipts.py \
  runs/commonbase-spatial runs/commonbase-object runs/commonbase-goal runs/commonbase-long
```

每个专家最少保留并传送以下文件，套件名字不能丢：

```text
FastWAM-handoff/
  common-base/
    base.json
    common-base.pt                 # 或 manifest 对应的实际共同权重
  spatial/
    selected.pt                    # 经你确认的所选训练 checkpoint
    config.yaml
    dataset_stats.json
    launch.json
    common-base-init-rank0.json
    common-base-init-rank1.json     # 按实际卡数传全部 rank 回执
    ...
  object/                          # 同样一组文件
  goal/                            # 同样一组文件
  long/                            # 同样一组文件
  SELECTION.md                     # 手动记录来源路径、原 step、SHA256、选择依据
  evaluation/                      # 有成功率评测时带原始结果/协议；未评测则明确写明
```

具体规则：

- 四个 `selected.pt` 来自各自 `checkpoints/weights/step_XXXXXX.pt`，**不是** `checkpoints/state/`。
- 可以保留原 step 文件名，不强制重命名；重命名时一定记录原文件名/步数。
- 不要把“最后一步”写成“最佳 checkpoint”，除非有对应评测结果。
- 四份 `dataset_stats.json` 都要传，不能只传一份覆盖其他套件。
- 所有 rank 回执要齐全；不能只取 rank0 来代替完整初始化审计。
- 同时提供代码版本（`git rev-parse HEAD`）和 `SOURCE_SHA256.json`。
- 本项目后续融合建议保留共同 base 本体。即使某个融合实现不需要它，也有助于核对参数起点。
- 若融合服务器没有相同冻结资源，还需传/下载第 2 节对应 assets；若已有且哈希一致，不必重复传。

**仅有四份专家权重还不足以保证 TCR 可直接融合**：TCR 还需要其实现所要求的校准观测/轨迹或特征缓存。
本仓库只负责共同 base 训练，不包含 TCR 校准采集/融合入口，也不会生成这些缓存。
校准数据应按实际使用的 TCR 管线另外准备，不能把 `dataset_stats.json` 当成校准样本。

### 保留原 run 目录的最小打包例子

假设：四个 run 都叫 `runs/commonbase-套件名`，
都明确选定 `step_012000.pt`，且共同起点使用 fresh Wan 方案。
**这些只是路径示例，不表示这些文件当前已经生成或该步数最好。**
在仓库根目录运行；任意文件不存在会停止：

```bash
python - <<'PY'
from pathlib import Path
import tarfile

paths = [Path("artifacts/common-base/base.json"),
         Path("artifacts/common-base/common-base.pt"),
         Path("SOURCE_SHA256.json")]
for suite in ("spatial", "object", "goal", "long"):
    run = Path("runs") / f"commonbase-{suite}"
    paths += [run / "checkpoints/weights/step_012000.pt",
              run / "config.yaml", run / "dataset_stats.json", run / "launch.json"]
    receipts = sorted(run.glob("common-base-init-rank*.json"))
    if not receipts:
        raise SystemExit(f"Missing initialization receipts: {run}")
    paths += receipts
for path in paths:
    if not path.is_file():
        raise SystemExit(f"Missing: {path}; correct the selected checkpoint paths first")
# x: refuses to overwrite an existing archive; weights usually compress poorly.
with tarfile.open("FastWAM-fusion-handoff.tar", "x") as archive:
    for path in paths:
        archive.add(path, arcname=str(path), recursive=False)
print("Created FastWAM-fusion-handoff.tar; add selection/evaluation records separately.")
PY
sha256sum FastWAM-fusion-handoff.tar
```

该例子只打包上述文件，不打包全部训练 state/数据；实际使用旧 policy 作为 base 时，
必须将例子中的 base 路径换成真正引用的文件，并一并交代引用关系。
推荐通过 SSH/SFTP/rsync 传到指定服务器，不上传 GitHub。
接收端复算 SHA256 对比，传输中断时不要把不完整归档当作有效结果。

## 4. 如果需要续训，再额外传这些

纯权重足够做 weights-only 新分支初始化，**不足以精确恢复旧训练**。
断点恢复需要该分支的：

- `checkpoints/state/step_XXXXXX/` **整个目录**，包括所有 rank 的优化器/模型/随机状态文件及 `trainer_state.json`。
- 原 `config.yaml`、`launch.json`、初始化回执，以及原数据、文本缓存、共同 base/资源。
- 相同软件版本、卡数、batch 和训练配置。

当前 resume 入口记录有原机器绝对路径；搬机器时必须保留相同路径映射，或逐项迁移校验
manifest 路径、数据路径、output_dir、state 路径后再做短恢复验收。
不能只把 state 拷到另一个随机目录就认为可以无缝 resume。

## 5. 文件大小与不必传的内容

- GitHub 代码是小文件，代码压缩包约百余 KB。
- Wan 下载资源约 34 GB；专家完整 checkpoint 和共同 base 也都是 GB 级，实际大小以文件为准。
- full-state 含优化器等，通常比单份纯权重更大；只做融合时无需传它。
- 不需要传所有中间 checkpoint、训练视频、完整旧日志或原始数据包，除非接收方明确需要。
- 整包大小取决于你选择的 checkpoint/state 数量，不把代码包大小当成权重传输大小。

```bash
du -sh assets data checkpoints artifacts/common-base runs
```

路径未生成时出现不存在提示是正常的。先选文件、看大小、校验，再传输。
