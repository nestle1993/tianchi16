from datetime import datetime
import cPickle
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--user_info_path',
                        help="user info path, ./dataset/user_view.txt " +
                        "./dataset/user_pay.txt or ./dataset/extra_user_view.txt",
                        default='')
    flags = parser.parse_args()

    data_path = flags.user_info_path

    shop_info_count = {id_ : {} for id_ in range(1, 2001)}
    f = open(data_path, "rb")
    lines = f.readlines()
    print("start process...")
    for i, line in enumerate(lines):
        slots = line.split(",")
        shop_id = int(slots[1])
        t = datetime.strptime(slots[2].strip("\n"), "%Y-%m-%d %H:%M:%S")
        day = slots[2].split(" ")[0]
        if day not in shop_info_count[shop_id]:
            # 0-24 hours, month, day, week number
            shop_info_count[shop_id][day] = [0] * 28
        # hour count
        shop_info_count[shop_id][day][t.hour] += 1
        # month note
        shop_info_count[shop_id][day][25] = t.month
        # day note
        shop_info_count[shop_id][day][26] = t.day
        # week number note
        shop_info_count[shop_id][day][27] = t.weekday()
    
        if i > 0 and i % 1000 == 0:
            print("finished %d" % i)
            
    if not os.path.exists("./pkl_files"):
        os.system("mkdir pkl_files")
    cPickle.dump(
        shop_info_count, open(
            "./pkl_files/%s_count.pkl" % data_path.split("/")[-1][:-4], "wb"))
        
    
    
