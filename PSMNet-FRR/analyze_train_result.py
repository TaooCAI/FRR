# -*- coding: utf-8 -*-
import os
import re

import torch
import matplotlib.pyplot as plt

model_dir = './model_save_g32_retrain_from_scratch_using_flyingthings3D_TEST'
# pattern_test_loss_file = r'(\d+)_test_loss.tar'
pattern_test_loss_file = r'test_(\d+)\.pth'
y_test = []
test_loss_file_list = [
    x for x in os.listdir(model_dir)
    if re.match(pattern_test_loss_file, x, re.IGNORECASE)
]
print(test_loss_file_list)
test_loss_file_list = sorted(
    test_loss_file_list,
    key=lambda x: int(re.findall(pattern_test_loss_file, x)[0]))

for loss_file in test_loss_file_list:
    loss_file = os.path.join(model_dir, loss_file)
    state = torch.load(loss_file, map_location='cpu')
    y_test.append(state['test_loss'])
    print('%s test loss: %.3lf' % (re.findall(
        pattern_test_loss_file, loss_file)[0], state['test_loss']))

# x = list(range(len(y_test)))
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.plot(x, y_test, label='test_loss')

# if os.path.exists(log_file):
#     train_and_eval_loss = torch.load(log_file, map_location='cpu')
#     y_eval = train_and_eval_loss['eval_loss_list']
#     x = list(range(len(y_eval)))
#     y_train = train_and_eval_loss['train_loss_list']
#     plt.plot(x, y_eval, label='eval_loss')
#     plt.xlabel('epoch')
#     plt.ylabel('loss')
#     # plt.plot(x, y_train, label='train_loss')

#     import pprint
#     pprint.pprint(train_and_eval_loss)

# plt.legend()
# plt.show()

# data = torch.load('datasets_relative_pathlist_tuple_except_unused_files.pth', map_location='cpu')
# data = data['data']
# import copy
# new_data = []
# for item in data:
#     new_item = []
#     for path in item:
#         if 'flying' in path.lower():
#             new_item.append(path)
#     new_data.append(new_item)
# state = {
#     'data':new_data
# }
# torch.save(state,'flying3d_relative_pathlist_tuple_except_unused_files.pth')