# 来源与修改范围

此包是本工作区已有 Fast-WAM 实现的独立快照，不是声称完全等同于 upstream main 的镜像。
原始代码来源：工作区中的 `vla-merge_table4/Fast-WAM/source/src/fastwam`，
以及其训练配置/预计算脚本。没有复制旧日志、权重、数据、机器专用的实验启动器。
这些路径只是来源记录，运行时不依赖原工作区。

上游：[yuantianyuan01/FastWAM](https://github.com/yuantianyuan01/FastWAM)。
查阅 README/许可时对应参考 commit 为
`7faa71108368fbb3b6885649f112af607427a2d4`；这不是所复制本地源码的可证明版本号。
MIT LICENSE 保留 FastWAM 作者署名；LeRobot 等第三方文件保留原许可头。
数据和外部权重受各自仓库条款约束，不因本代码包使用 MIT 就改为 MIT。

本包增加/调整：

- 固定初始化 seed 在模型构造前执行。
- 完整 `mot` 和状态编码器的严格加载，SHA256、有限值、逐张量回读、完整模型指纹检查。
- 跨 rank 和四专家初始化回执；封存时记录外部冻结资源与配置哈希。
- Wan 组件加载也改为 strict=True，缺键不能悄悄留下随机参数；实际大模型兼容性仍待封存验证。
- 官方原始 VAE 路径，固定 revision 的下载器、安全解压、独立 DeepSpeed 配置。
- 新建训练和 full-state resume 分离，默认预览、已有输出拒绝覆盖。
- 独立打包布局、CPU 回归测试、训练手册。

没有宣称修改 TCR 算法、修复已经训练好的四个专家、或提高模型成功率。
标准训练流程仅依赖官方 Wan 预训练资源，不依赖本项目微调 checkpoint。

依赖选择参考：[PyTorch 官方历史安装命令](https://pytorch.org/get-started/previous-versions/)、
[TorchCodec 官方兼容表](https://github.com/meta-pytorch/torchcodec#compatibility-with-torch-versions)。
PyTorch 2.7.1 / torchvision 0.22.1 与 TorchCodec 0.5 的版本线已核对，
但该组合的本仓库完整 GPU 训练尚未验证。

`SOURCE_SHA256.json` 是交付代码内容清单，不包含自身、Git 元数据、缓存及大资源。
打包前运行 `python scripts/source_manifest.py` 生成；后续用
`python scripts/source_manifest.py --check` 检查交付文件是否改变。
