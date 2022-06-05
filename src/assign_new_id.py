import config
import pandas as pd

# read train and test ids
train_ids = pd.read_csv(config.TRAIN_TARGET, usecols=[0])
test_ids = pd.read_csv(config.TEST_TARGET, usecols=[0])
ids = pd.concat([train_ids, test_ids], ignore_index=True)

# remove duplicate and assign new id
ids = ids.drop_duplicates()
ids.reset_index(inplace=True)
ids.rename(columns={"index": "new_ID"}, inplace=True)
ids["new_ID"] = ids["new_ID"].astype("int32")

# write csv
ids.to_csv(config.NEW_IDS, header=True, index=False)
