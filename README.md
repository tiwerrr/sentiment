# 多模态情感分析
当代人工智能第五次实验作业

## 环境配置
python == 3.8.13  
pytorch == 1.12.1  
orchvision == 0.13.1  
transformers == 4.30.2  
pandas == 1.4.3  

## 文件结构
|-- data # 原数据集  
|-- token # 文本编辑器（放在本仓库的release处）  
|-- requirements.txt # 环境配置文件  
|-- main.py # 主函数，用于模型建构、训练、预测  
|-- prepare.ipynb # 数据处理文件  
|-- train.json # 运行prepare.ipynb后产生的训练数据  
|-- val.json # 运行prepare.ipynb后产生的验证数据  
|-- test.json # 运行prepare.ipynb后产生的测试数据  
|-- predict.txt # 预测文件  

## 运行
由于train.json、val.json、test.json在文件夹中已给出，直接运行main.py即可  
即输入python main.py（默认进行多模态融合模型训练，学习率为5e-5。若需修改模型和超参数，输入python main.py --0 --1e-5等即可）  
