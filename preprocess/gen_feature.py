import datetime
import pickle
pay_count = pickle.load(open("./pkl_files/user_pay_count.pkl", "rb"))
view_count = pickle.load(open("./pkl_files/merged_user_view_count.pkl", "rb"))

all_shop_feature = []
d_start = datetime.datetime(2015, 7, 1)
d_end = datetime.datetime(2016, 11, 1)
d_omit = datetime.datetime(2015, 12, 12)
for shop_id in range(1, 2001):
    print("now shop %d" % shop_id)
    shop_feature = []
    d = d_start
    while d != d_end:
        if d == d_omit:
            d += datetime.timedelta(1)
            continue
        d_str = d.strftime("%Y-%m-%d")
        if d_str not in pay_count[shop_id]:
            pay_feature = [0] * 28
        else:
            pay_feature = pay_count[shop_id][d_str]
        if d_str not in view_count[shop_id]:
            view_feature = [0] * 28
        else:
            view_feature = view_count[shop_id][d_str]
        shop_feature.append(
            pay_feature[:25] + \
            view_feature[:25] + \
            [d.month, d.day, d.weekday()]
        )
        d += datetime.timedelta(1)
    all_shop_feature.append(shop_feature)

pickle.dump(all_shop_feature, open("./pkl_files/shop_feature.pkl", "wb"))
    
