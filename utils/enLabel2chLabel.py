import torch
 
# 加载模型
w=torch.load(r'C:\Users\Administrator\ultralytics\runs\segment\train_yolov11seg_demo1\weights\best.pt')
 
#打印所有name
print(w.get('model').names)
 
# 定义一个将英文单词映射到中文单词的字典
word_map = {
    'crack': '裂缝',
    'spalling': '剥落',
}
 
# 遍历列表，将每个英文单词替换为其中文对应词
for i in range(len(w.get('model').names)):
    if w.get('model').names[i] in word_map:
        w.get('model').names[i] = word_map[w.get('model').names[i]]
 
# 打印替换后的列表
print('替换后')
print(w.get('model').names)
#保存替换后的模型
torch.save(w,'./outputs/new_model.pt')