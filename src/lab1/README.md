# AlexNet on MNIST

实验通过构建一个AlexNet来进行MNIST手写数字识别任务。并在实验中进行以下观察任务：

- 检查前向传播的正确性：没有训练随机初始化的情况下，损失接近 $\ln 10 \approx 2.302$。 
- 查看网络参数。
- 在不同学习率下训练网络，range：`0.00001, 0.0001, 0.001, 0.01`，并计算每个epoch的`train loss, train loss, test acc`。 
- 绘制每个lr下的`train/test loss vs batch`，`train/test acc vs batch`(其中，为了减少计算，对于test我们在每个epoch的额前500个batch每50次测试一次)。 
- 可视化各个lr下的第一层的大核卷积，形状为`(64, 11)`。  
- 绘制各个lr下的混淆矩阵，用于查看错例。  
