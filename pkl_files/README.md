### The preprocessing results will be dumped here:
- info_dict.pkl:
    - list of dict of `shop_id, city, location_id, per_pay, score, comment_cnt, shop_level, cate_1_name, cate_2_name, cate_3_name`: index. The map of string/integer to index.
    - e.g: 
    ```
    [{...},
     {'北京':0, '上海': 1, ...},
     {...},
     {'1': 0, '2': 1, ...},
     {...},
     ...
    ]
    ```

- shop_label.pkl:
     - labeled shop info. Replace slots exclude `shop_id` and `location_id` to their mapping indexes.
     - e.g: 
    ```
    [[1, 49, 885, 19, 5, 3, 2, 2, 15, 30],
     [2, 56, 64, 8, 0, 0, 0, 1, 2, 5],
     [3, 96, 774, 14, 3, 14, 1, 2, 15, 25],
      ...,
     [2000, 9, 378, 16, 3, 14, 1, 2, 16, 4]
    ]
    ```

- user_pay_count.pkl:
    - dict of {shop_id: daily count}
        - daily count: dict of {day: features}
            - features: [0-24 hour pay total count] + [month, day, week]
    - e.g:
    ```
    {1: {'2016-06-18': [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 3, 2, 2, 0, 0, 2, 0, 0, 2, 1, 1, 0, 0, 6, 18, 5], '2016-06-19': [...],...},
     2: {'2016-06-18': [...], '2016-06-19': [...], ...},
     ...
     2000: {'2016-06-18': [...], '2016-06-19': [...], ...}
    }
    ```

- user_view_count.pkl:
    - the same format as above.

- extra_user_view_count.pkl:
    - the same format as above.

