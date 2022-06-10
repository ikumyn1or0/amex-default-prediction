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
    data["D_63"] = data["D_63"].replace({np.nan: -1, "CR": 0, "XZ": 1, "XM": 2, "CO": 3, "CL": 4, "XL": 5}).astype(np.int8)
    data["D_64"] = data["D_64"].replace({np.nan: -1, "O": 0, "-1": 1, "R": 2, "U": 3}).astype(np.int8)
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
    data["D_82"] = np.round(data["D_82"]*2).fillna(-1).astype(np.int8)
    data["D_83"] = np.round(data["D_83"]).fillna(-1).astype(np.int8)
    data["D_84"] = np.round(data["D_84"]*2).fillna(-1).astype(np.int8)
    data["D_86"] = np.round(data["D_86"]).fillna(-1).astype(np.int8)
    data["D_87"] = np.round(data["D_87"]).fillna(-1).astype(np.int8)
    data["D_89"] = np.round(data["D_89"]*9).fillna(-1).astype(np.int8)
    data["D_90"] = np.round(data["D_90"]*2).fillna(-1).astype(np.int8)
    data["D_92"] = np.round(data["D_92"]).fillna(-1).astype(np.int8)
    data["D_93"] = np.round(data["D_93"]).fillna(-1).astype(np.int8)
    data["D_94"] = np.round(data["D_94"]).fillna(-1).astype(np.int8)
    data["D_96"] = np.round(data["D_96"]).fillna(-1).astype(np.int8)
    data["D_103"] = np.round(data["D_103"]).fillna(-1).astype(np.int8)
    data["D_106"] = np.round(data["D_106"]*23).fillna(-1).astype(np.int16)
    data["D_107"] = np.round(data["D_107"]*3).fillna(-1).astype(np.int8)
    data["D_108"] = np.round(data["D_108"]).fillna(-1).astype(np.int8)
    data["D_109"] = np.round(data["D_109"]).fillna(-1).astype(np.int8)
    data["D_111"] = np.round(data["D_111"]*2).fillna(-1).astype(np.int8)
    data["D_112"] = np.round(data["D_112"]).fillna(-1).astype(np.int8)
    data["D_113"] = np.round(data["D_113"]*5).fillna(-1).astype(np.int8)
    data["D_114"] = np.round(data["D_114"]).fillna(-1).astype(np.int8)
    data["D_116"] = np.round(data["D_116"]).fillna(-1).astype(np.int8)
    data["D_117"] = np.round(data["D_117"]+1).fillna(-1).astype(np.int8)
    data["D_120"] = np.round(data["D_120"]).fillna(-1).astype(np.int8)
    data["D_122"] = np.round(data["D_122"]*8).fillna(-1).astype(np.int8)
    data["D_123"] = np.round(data["D_123"]).fillna(-1).astype(np.int8)
    data["D_124"] = np.round(data["D_124"]*22+1).fillna(-1).astype(np.int8)
    data["D_125"] = np.round(data["D_125"]).fillna(-1).astype(np.int8)
    data["D_126"] = np.round(data["D_126"]+1).fillna(-1).astype(np.int8)
    data["D_127"] = np.round(data["D_127"]).fillna(-1).astype(np.int8)
    data["D_128"] = np.round(data["D_128"]).fillna(-1).astype(np.int8)
    data["D_129"] = np.round(data["D_129"]).fillna(-1).astype(np.int8)
    data["D_130"] = np.round(data["D_130"]).fillna(-1).astype(np.int8)
    data["D_135"] = np.round(data["D_135"]).fillna(-1).astype(np.int8)
    data["D_136"] = np.round(data["D_136"]*4).fillna(-1).astype(np.int8)
    data["D_137"] = np.round(data["D_137"]).fillna(-1).astype(np.int8)
    data["D_138"] = np.round(data["D_138"]*2).fillna(-1).astype(np.int8)
    data["D_139"] = np.round(data["D_139"]).fillna(-1).astype(np.int8)
    data["D_140"] = np.round(data["D_140"]).fillna(-1).astype(np.int8)
    data["D_143"] = np.round(data["D_143"]).fillna(-1).astype(np.int8)
    data["D_145"] = np.round(data["D_145"]*11).fillna(-1).astype(np.int8)
    data["R_2"] = np.round(data["R_2"]).fillna(-1).astype(np.int8)
    data["R_3"] = np.round(data["R_3"]*10).fillna(-1).astype(np.int8)
    data["R_4"] = np.round(data["R_4"]).fillna(-1).astype(np.int8)
    data["R_5"] = np.round(data["R_5"]*2).fillna(-1).astype(np.int8)
    data["R_8"] = np.round(data["R_8"]).fillna(-1).astype(np.int8)
    data["R_9"] = np.round(data["R_9"]*6).fillna(-1).astype(np.int8)
    data["R_10"] = np.round(data["R_10"]).fillna(-1).astype(np.int8)
    data["R_11"] = np.round(data["R_11"]*2).fillna(-1).astype(np.int8)
    data["R_13"] = np.floor(data["R_13"]*31).fillna(-1).astype(np.int8)
    data["R_15"] = np.round(data["R_15"]).fillna(-1).astype(np.int8)
    data["R_16"] = np.round(data["R_16"]*2).fillna(-1).astype(np.int8)
    data["R_17"] = np.floor(data["R_17"]*35).fillna(-1).astype(np.int8)
    data["R_17"] = np.floor(data["R_17"]*35).fillna(-1).astype(np.int8)
    data["R_19"] = np.floor(data["R_19"]).fillna(-1).astype(np.int8)
    data["R_19"] = np.floor(data["R_19"]).fillna(-1).astype(np.int8)
    data["R_20"] = np.round(data["R_20"]*10).fillna(-1).astype(np.int16)
    data["R_21"] = np.round(data["R_21"]).fillna(-1).astype(np.int8)
    data["R_22"] = np.round(data["R_22"]).fillna(-1).astype(np.int8)
    data["R_23"] = np.round(data["R_23"]).fillna(-1).astype(np.int8)



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

