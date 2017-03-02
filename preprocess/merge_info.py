import pickle
view_count = pickle.load(open("./pkl_files/user_view_count.pkl", "rb"))
extra_view_count = pickle.load(open("./pkl_files/extra_user_view_count.pkl", "rb"))

not_in_count = 0
# merge view_count and extra_view_count
for shop_id, info in extra_view_count.items():
    for day, feature in info.items():
        if day not in view_count[shop_id]:
            view_count[shop_id][day] = feature
            not_in_count += 1
            print("day %s not in shop_id %d!" % (day, shop_id))
        else:
            for i in range(25):
                view_count[shop_id][day][i] += feature[i]

print("total not in count:", not_in_count)
pickle.dump(view_count, open("./pkl_files/merged_user_view_count.pkl", "wb"))

