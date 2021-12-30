import pandas as pd
import numpy as np


def prefilter_items(data, take_n_popular=5000, item_features=None, fake_item=999999, max_price=50, min_price=1):
    data = data.copy()
    # Уберем не интересные для рекоммендаций категории (department)
    if item_features is not None:
        department_size = pd.DataFrame(item_features. \
                                       groupby('department')['item_id'].nunique(). \
                                       sort_values(ascending=False)).reset_index()

        department_size.columns = ['department', 'n_items']
        rare_departments = department_size[department_size['n_items'] < 150].department.tolist()
        items_in_rare_departments = item_features[
            item_features['department'].isin(rare_departments)].item_id.unique().tolist()

        data = data[~data['item_id'].isin(items_in_rare_departments)]

    # Уберем слишком дешевые товары (на них не заработаем). 1 покупка из рассылок стоит 60 руб.
    data['price'] = data['sales_value'] / (np.maximum(data['quantity'], 1))
    data = data[data['price'] > min_price]

    # Уберем слишком дорогие товары
    data = data[data['price'] < max_price]

    # Возбмем топ по популярности
    popularity = data.groupby('item_id')['quantity'].sum().reset_index()
    popularity.rename(columns={'quantity': 'n_sold'}, inplace=True)

    top = popularity.sort_values('n_sold', ascending=False).head(take_n_popular).item_id.tolist()

    # Заведем фиктивный item_id (если юзер покупал товары из топ-5000, то он "купил" такой товар)
    data.loc[~data['item_id'].isin(top), 'item_id'] = fake_item
    
    return data

def print_stats_data(df_data, name_df, item_col='item_id', user_col='user_id'):
    print(name_df)
    print(f"Shape: {df_data.shape} Users: {df_data[user_col].nunique()} Items: {df_data[item_col].nunique()}")

def get_hots_users(first_df, second_df):
    hot_users = set(first_df.user_id.value_counts().index.to_list()).intersection(set(second_df.user_id.value_counts().index.to_list()))
    return first_df.loc[first_df.user_id.isin(hot_users)], second_df.loc[second_df.user_id.isin(hot_users)]