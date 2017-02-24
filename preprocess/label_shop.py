import cPickle

# shop_id, city, location_id, per_pay, score, comment_cnt,
# shop_level, cate_1_name, cate_2_name, cate_3_name
info = [[], [], [], [], [], [], [], [], [], []]

# keep labeled shop info
shop_label = []

f = open("./dataset/shop_info.txt", "rb")
lines = f.readlines()
for line in lines:
    slots = line.split(",")
    for i, it in enumerate(slots):
        info[i].append(it)

def gen_dict(l):
    l = set(l)
    return dict(zip(l, range(len(l))))

# generate dict
info = [gen_dict(l) for l in info]

print("-----kind count-----")
for l in info:
    print(len(l))

for line in lines:
    slots = line.split(",")
    labeled = [info[i][it] if i != 0 and i != 2 else int(it)
               for i, it in enumerate(slots)
              ]
    shop_label.append(labeled)

cPickle.dump(info, open("./pkl_files/info_dict.pkl", "wb"))
cPickle.dump(shop_label, open("./pkl_files/shop_label.pkl", "wb"))
