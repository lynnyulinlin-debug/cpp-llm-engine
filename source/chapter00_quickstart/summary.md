# 00.7 Chapter Wrap-up 本章解决了什么

这章回答的是：本地推理基线是否真的已经跑通？

如果你已经能够在本地跑通 `llama.cpp`，并完成一次最小推理验证，说明这条基线已经成立。可选的 `vLLM` 部署只用于体验真实 serving baseline，不影响后续主线继续学习。

## 本章解决了什么

- 第一条真实推理链路已经跑通
- 环境、编译链路和模型下载已经连通
- 教程起跑线已经固定下来
- 如果完成可选小节，也已经见过 OpenAI-compatible server 的最小服务形态

## 为什么重要

如果没有这一步，后面的 `Tensor`、`ModelLoader` 和算子章节都只能停留在纸面推导。`chapter00` 先把“跑通”这件事固定住，`chapter01` 再把“读入”接上，起步层才算闭环。

## 下一步

进入 `chapter01_tensor_loader`，把权重安全地读进程序，并把输入对象模型先定下来。

## 复盘路径

- 先看 <a href="../_code/chapter00_quickstart/main_quick.cpp"><code>main_quick.cpp</code></a> 和 <a href="../_code/chapter00_quickstart/run.sh"><code>run.sh</code></a>
- 再看 [`00.2 Clone and Build`](./02_clone_and_build_llamacpp.md) 和 [`00.4 First Inference`](./04_first_inference.md)
- 可选回看 [`00.6 Deploy vLLM`](./06_deploy_vllm.md)
- 如果要复盘整章，回到 [`chapter00 导读`](./intro.md)
