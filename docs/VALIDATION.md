# 验证记录 — 2026-09-24

## 本次实际通过

- 27 项 CPU 单元/集成边界测试，全数通过。
- 真正的共同 checkpoint 严格读写使用小模型测试，不只是字符串检查。
- 真实 runtime 函数的初始化顺序验证（重模型/训练器用测试替身）：构造 → 加载 → 建优化器 → 训练；
  封存指纹不匹配时，在建优化器/启动训练前拒绝。
- SHA256 不匹配、缺失状态编码器/主干键、形状不符、NaN、旧格式 checkpoint 均被拒绝。
- rank 身份不一致、四分支不一致、重复写回执和覆盖旧 run 被拒绝。
- fresh seal 小模型回读与 manifest 校验、四套件命令共享同一 base、resume 与初始化分离。
- 真正 Hydra compose/resolve 验证 action_dim=7、proprio_dim=8、
  Long 数据映射以及训练覆盖参数；不需要下载或实例化大模型。
- 安全解压的路径穿越/链接拒绝测试。
- 从其他 cwd 执行 workflow 预览，无训练目录创建。
- 压缩包解压到独立临时目录后复测；这一检查曾发现忽略规则误排除
  configs/data，已改为只忽略根目录 /data/，并新增回归测试。
- 安装 shell 语法、下载/准备/封存的预览命令。
- setuptools wheel 构建成功（仅构建，不安装依赖），源码可独立打包。
- 实际 Fast-WAM runtime 和模型模块在已有环境中成功导入。

CPU 测试环境：Python 3.10、Torch 2.2.0+cu121，CUDA 隐藏。
模块导入环境：Python 3.12、Torch 2.7.0a0 NVIDIA 25.03、
Transformers 4.51.3、工作区已有依赖 overlay。
**这两者都不是 README 的全新 Python 3.10 / Torch 2.7.1+cu128 安装验证。**

现有导入环境执行 doctor 返回 not ready：未发现 TorchCodec，PATH 中未发现 FFmpeg。
这不是已跑通新环境的证据；安装脚本列出了 TorchCodec，FFmpeg 需要用户先准备。

## 未完成、不可声称通过

- 在全新环境执行安装并完成 pip check。
- 下载锁定的全部大资源、用官方原始 VAE 成功加载整个 Fast-WAM。
- 使用真实完整 checkpoint 执行 seal 与 GPU 训练加载的指纹比对。
- 真正四卡 DeepSpeed 的前向/反向、优化器更新、数值稳定性和显存峰值。
- 真正训练状态中断/恢复、真实数据解码及完整文本缓存覆盖。
- 四个新专家训练、LIBERO rollout 成功率、TCR/Soup 融合效果。

本次测试未下载大资源、未创建真实共同 base、未占用 GPU 训练，
未覆盖既有权重/训练目录。
代码层面测试通过不等于算法效果已经验证；共同初始化也不是 TCR 必然提升的保证。

## 使用机器上的验收顺序

1. 安装 → `pip check` → `doctor.py`，先解决依赖缺失。
2. 固定版本下载 → 安全解压 → ActionDiT 和文本缓存准备。
3. 官方 Wan 预训练权重 → seal → 保留 manifest 和完整共同权重。
4. 小规模多卡训练：至少两步 loss 有限、所有 rank 回执齐全、weights 和 state 保存成功。
5. 独立短 run 的中断/恢复验收，确认状态继续而非重新初始化。
6. 再启动四套件正式训练，回执比较 accepted=true，最后另做策略评测和融合。

迁移复核：

```bash
python scripts/source_manifest.py --check
python -m unittest discover -s tests -v
```

运行目录中 `launch.json` 是计划及共同 base 记录，
`common-base-init-rank*.json` 才是模型实际加载的证据；
二者都不是训练完成或成功率证据。
