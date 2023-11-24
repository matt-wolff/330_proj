import pandas as pd
SEED = 42

df = pd.read_csv('data/go_tasks.csv', encoding='utf-8')
num_tasks, _ = df.shape  # Note: _ = 3.
df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
train_df = df[:int(0.7*num_tasks)]
train_df.to_csv('data/train_go_tasks.csv', index=False)
val_df = df[int(0.7*num_tasks):int(0.8*num_tasks)]
val_df.to_csv('data/val_go_tasks.csv', index=False)
test_df = df[int(0.8*num_tasks):]
test_df.to_csv('data/test_go_tasks.csv', index=False)
