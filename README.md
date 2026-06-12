# MNIST手写数字分类 & TensorBoard可视化
## 项目介绍
基于PyTorch搭建轻量卷积神经网络，完成MNIST手写数字十分类任务；使用TensorBoard实现训练过程全维度可视化，监控损失、准确率、样本与网络结构。

## 技术栈
Python / PyTorch / torchvision / TensorBoard / CNN

## 运行方式
1. 环境依赖：`pip install -r requirements.txt`
2. 启动训练：`python train.py`
3. 可视化面板：`tensorboard --logdir=logs`

## 核心功能与亮点
1. 完整深度学习流水线：数据加载 -> CNN模型构建 -> 反向传播训练 -> 测试集评估
2. TensorBoard多维日志记录：
   - 标量曲线：训练/验证损失、验证准确率
   - 图像：训练集样本预览
   - 计算图：CNN网络结构可视化
3. 模块化代码结构，可快速迁移至其他图像分类任务
