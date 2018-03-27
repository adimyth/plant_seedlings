import matplotlib.pyplot as plt
import pandas as pd

dataframe = pd.read_csv("attempt5.csv", sep='\t')

# accuracy plot
# plt.plot(dataframe['acc'])
# plt.plot(dataframe['val_acc'])
# plt.title("Model Accuracy")
# plt.ylabel("accuracy")
# plt.xlabel("epoch")
# plt.legend(['train', 'validation'], loc='best')
# plt.savefig('accuracy_5.png')

# loss plot
plt.plot(dataframe['loss'])
plt.plot(dataframe['val_loss'])
plt.title("Model Accuracy")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(['train', 'validation'], loc='best')
plt.savefig('loss_5.png')