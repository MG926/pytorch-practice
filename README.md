# PyTorch 学习路线

本目录按照从 Tensor 到完整项目的顺序组织。每份 Notebook 都包含 TODO 和自动检查；完成当前模块的“过关标准”后再进入下一模块。

| 顺序 | 模块 | Notebook | 核心目标 |
|---|---|---|---|
| 01 | Tensor | `01 tensor practice/tensor_exercises.ipynb` | 创建、索引、形状、广播、设备 |
| 02 | 自动求导 | `02 autograd practice/autograd_exercises.ipynb` | 计算图、反向传播、梯度管理 |
| 03 | 训练循环 | `03 training loop practice/training_loop_exercises.ipynb` | 损失、优化器、epoch、训练/评估模式 |
| 04 | 数据加载 | `04 dataset and dataloader practice/dataset_dataloader_exercises.ipynb` | Dataset、DataLoader、划分、批处理 |
| 05 | 神经网络 | `05 neural network practice/neural_network_exercises.ipynb` | nn.Module、Sequential、初始化、残差 |
| 06 | 分类任务 | `06 classification practice/classification_exercises.ipynb` | logits、交叉熵、指标、分类训练 |
| 07 | 保存加载 | `07 model saving and loading/model_saving_loading_exercises.ipynb` | state_dict、checkpoint、恢复训练 |
| 08 | 调试 | `08 debugging practice/debugging_exercises.ipynb` | shape、dtype、device、梯度与数值稳定性 |
| 09 | 综合项目 | `09 mini projects/mini_projects.ipynb` | 回归、非线性分类、保存与推理 |

## 建议节奏

每次学习一个知识点后先完成对应练习，不要直接查看完整答案。自动检查失败时，先打印相关 Tensor 的 shape、dtype、device 和数值。完成 Notebook 后记录：

- 哪些题一次通过。
- 哪些题需要查文档。
- 错误的原因是什么。
- 如果重新写一次，是否能脱离提示完成。

## 环境

在 PowerShell 中进入项目并激活环境：

```powershell
Set-Location "F:\pytorch practice"
.\.venv\Scripts\Activate.ps1
```

在 VS Code 或 Jupyter 中选择 `.venv\Scripts\python.exe` 作为 Notebook 内核。

