import config
import numpy as np
import pandas as pd

def preprocess(data, ids):
    data["B_4"] = np.floor(data["B_4"]*78).fillna(-1).astype(np.int16)
    data["B_8"] = np.round(data["B_8"]).fillna(-1).astype(np.int8)
    data["B_9"] = np.floor(data["B_9"]*160).fillna(-1).astype(np.int16) # doubtful
    data["B_16"] = np.round(data["B_16"]*12).fillna(-1).astype(np.int8)
    # B_18: doubtful
    data["B_19"] = np.floor(data["B_19"]*100).fillna(-1).astype(np.int8)
    data["B_20"] = np.round(data["B_20"]*17).fillna(-1).astype(np.int8)
    data["B_22"] = np.round(data["B_22"]*2).fillna(-1).astype(np.int8)
    # B_27: doubtful
    data["B_30"] = np.round(data["B_30"]).fillna(-1).astype(np.int8)
    data["B_31"] = np.round(data["B_31"]).fillna(-1).astype(np.int8)
    data["B_32"] = np.round(data["B_32"]).fillna(-1).astype(np.int8)
    data["B_33"] = np.round(data["B_33"]).fillna(-1).astype(np.int8)
    data["B_38"] = np.round(data["B_38"]).fillna(-1).astype(np.int8)
    data["B_41"] = np.round(data["B_41"]).fillna(-1).astype(np.int8)
    data["D_39"] = np.round(data["D_39"]*34).fillna(-1).astype(np.int8)
    data["D_44"] = np.round(data["D_44"]*8).fillna(-1).astype(np.int8)
    # D_45: doubtful
    data["D_49"] = np.round(data["D_49"]*71).fillna(-1).astype(np.int8)
    data["D_51"] = np.round(data["D_51"]*3).fillna(-1).astype(np.int8)
    data["D_59"] = np.round(data["D_59"]*48+4).fillna(-1).astype(np.int8)
    # D_62: doubtful
    data["D_63"] = data["D_63"].apply(lambda t: {np.nan: -1, "CR": 0, "XZ": 1, "XM": 2, "CO": 3, "CL": 4, "XL": 5}[t]).astype(np.int8)
    data["D_64"] = data["D_64"].apply(lambda t: {np.nan: -1, "O": 0, "-1": 1, "R": 2, "U": 3}[t]).astype(np.int8)
    data["D_65"] = np.floor(data["D_65"]*38).fillna(-1).astype(np.int16)
    data["D_68"] = np.round(data["D_68"]).fillna(-1).astype(np.int8)
    data["D_70"] = np.round(data["D_70"]*4).fillna(-1).astype(np.int8)
    data["D_72"] = np.round(data["D_72"]*3).fillna(-1).astype(np.int8)
    data["D_74"] = np.round(data["D_74"]*14).fillna(-1).astype(np.int8)
    data["D_75"] = np.round(data["D_75"]*15).fillna(-1).astype(np.int8)
    data["D_78"] = np.round(data["D_78"]*2).fillna(-1).astype(np.int8)
    data["D_79"] = np.round(data["D_79"]*2).fillna(-1).astype(np.int8)
    data["D_80"] = np.round(data["D_80"]*5).fillna(-1).astype(np.int8)
    data["D_81"] = np.round(data["D_81"]).fillna(-1).astype(np.int8)



    return data


# read id
ids = pd.read_csv(config.NEW_IDS)
ids["new_ID"] = ids["new_ID"].astype(np.int32)

# read train
train = preprocess(pd.read_csv(config.TRAIN_DATA, nrows=0), ids)
train_reader = pd.read_csv(config.TRAIN_DATA, chunksize=10000)
for i, tr in enumerate(train_reader):
    if i < 2:
        train = pd.concat([train, preprocess(tr, ids)], ignore_index=True)
    else:
        break
print(train["B_8"])
