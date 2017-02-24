### scripts for preprocessing
- label_shop.py
    - map shop infos to indexes, and translate original shop infos to corresponding indexes.
    - at root, RUN `python preprocess/label_shop.py`
    - outputs: 
        - ./pkl_files/info_dict.pkl and ./pkl_files/shop_label.pkl

- count_user_info.py
    - count daily view/pay numbers for every shop.
    - at root, RUN `python preprocess/count_user_info.py -p ${TXT_FILE_PATH}`. e.g `python preprocess/count_user_info.py -p ./dataset/user_view.txt`
    - outputs:
        - ./pkl_files/${TXT_FILE_NAME}_count.pkl
        - see pkl_files/README.md for detail format of outputs.
