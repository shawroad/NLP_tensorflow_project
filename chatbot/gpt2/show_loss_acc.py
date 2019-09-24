"""

@file  : show_loss_acc.py

@author: xiaolu

@time  : 2019-09-23

"""
import json
import matplotlib.pyplot as plt


data = json.load(open('acc_loss.json', 'r'))
print(data)
plt.subplot('121')
plt.plot(range(100), data[0], c='gold')
plt.ylabel('loss')
plt.xlabel('epoch')
# plt.legend('loss')
plt.subplot('122')
plt.plot(range(100), data[1], c='red')
# plt.legend('acc')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.show()

