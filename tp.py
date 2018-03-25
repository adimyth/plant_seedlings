directories = ['Shepherds Purse', 'Common wheat', 'Sugar beet', 'Maize', 'Common Chickweed', 'Black-grass', 'Loose Silky-bent', 'Fat Hen', 'Cleavers', 'Scentless Mayweed', 'Small-flowered Cranesbill', 'Charlock']
import pandas as pd
df = pd.read_csv('predictions.csv')
# df['file'] = df['file'].str.replace(r'.png$', '')

column = df['species']
answers = []
for i in column:
    answers.append(directories[column[i]])
answers = pd.Series(answers)
df['species'] = answers
df.to_csv("answers.csv", index=False)

# import pandas as pd
# df = pd.read_csv('predictions.csv')
# print(list(df['species']))