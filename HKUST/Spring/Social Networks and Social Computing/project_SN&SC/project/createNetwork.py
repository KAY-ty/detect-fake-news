import networkx as nx
import json
import os
import csv
import time
import pickle

def preprocess():
    # path = os.path.dirname(os.getcwd())+"/yelp_dataset/yelp_academic_dataset_user.json"
    # print(path)

    file = open('yelp_academic_dataset_user.txt')
    temp = open('user.txt', 'w')
    pre = "["
    for line in file.readlines():
        if(pre != ""):
            print(pre)
            temp.write(pre)
        l = list(line)
        l.append(',')
        line = ''.join(l)
        pre = line

    l = list(pre)
    l.pop()
    l.append(']')
    pre = ''.join(l)

    temp.write(pre)


    # with open("temp.json", "r", encoding='utf8') as f:
    #     user_dict = json.load(f)

def create_user_graph():
    csv.field_size_limit(500*1024*1024)
    with open('yelp_academic_dataset_user.csv', 'r') as f:
        reader = csv.reader(f)
        print(type(reader))

        user_graph = nx.Graph()
        user_dict = {'user_id': {'name': 'user name', 'average_star': 'averange star of the user',
                         'friends_num': 'the number of friends', 'fans_num': 'the number of fans'}}

        i = 0
        for row in reader:
            friends = row[7]
            id = row[14]
            fans = row[8]
            averange_star = row[13]
            fans = row[8]
            name = row[19]
            print(i)
            i = i+1
            user_graph.add_node(id)
            user_dict[id] = {'name': name, 'average_star': averange_star,
                         'friends_num': len(friends.split(',')), 'fans_num': len(fans.split(','))}
            for friend in friends.split(','):
                user_graph.add_node(friend)
                user_graph.add_edge(id, friend)

    return user_graph, user_dict

def create_user_item_graph():
    csv.field_size_limit(500 * 1024 * 1024)
    with open('yelp_academic_dataset_review.csv', 'r') as f:
        reader = csv.reader(f)
        user_item_graph = nx.Graph()
        i = 0
        for row in reader:
            print(i)
            i = i+1
            user_id = row[7]
            business_id = row[4]
            star = row[5]
            user_item_graph.add_node(user_id)
            user_item_graph.add_node(business_id)
            user_item_graph.add_edge(user_id, business_id, weight = star)


    return user_item_graph

def test_pickle():
    file_name = "toy_dataset.pickle"
    with open(file_name, 'rb') as f:
        history_u_lists, history_ur_lists, history_v_lists, history_vr_lists, train_u, train_v, train_r, test_u, test_v, test_r, social_adj_lists, ratings_list = pickle.load(f)
        # print("history_u_lists: "+str(type(history_u_lists)))
        # print("history_ur_lists: "+str(type(history_ur_lists)))
        # print("train_u: "+str(type(train_u)))
        # print(len(train_u))
        # print(len(train_v))
        # print(len(train_r))
        # print(ratings_list)
        # print(type(ratings_list))
        # print(len(history_v_lists))
        # print(len(history_u_lists))
        print(test_r)
        dic = {0.5:0, 1.0:0, 1.5:0, 2.0:0, 2.5:0, 3.0:0, 3.5:0, 4.0:0}
        for i in train_r:
            dic[i] = dic[i]+1
        print(dic)


def create_pickle():
    csv.field_size_limit(500 * 1024 * 1024)
    with open('yelp_academic_dataset_user.csv', 'r') as f:
        social_adj_lists = {}
        reader = csv.reader(f)
        i = 0
        for row in reader:
            print("user"+str(i))
            i = i+1
            friends = row[7]
            id = row[14]
            if id == 'user_id':
                continue
            social_adj_lists[id] = friends
            # if i == 10000:
            #     break
            # break

    bytes_out = pickle.dumps(social_adj_lists)
    n_bytes = 2 ** 31
    max_bytes = 2 ** 31 - 1
    with open('social_list.pickle', 'wb') as f_out:
        for idx in range(0, len(bytes_out), max_bytes):
            f_out.write(bytes_out[idx:idx + max_bytes])
    # with open('social_list.pickle', 'wb') as f:
    #     pickle.dump(social_adj_lists, f)

    with open('yelp_academic_dataset_review.csv', 'r') as f:
        reader = csv.reader(f)
        history_u_lists = {}
        history_ur_lists = {}
        history_v_lists = {}
        history_vr_lists = {}
        train_u = []
        train_v = []
        train_r = []
        i = 0
        for row in reader:
            print("review"+str(i))
            i = i+1
            user_id = row[7]
            business_id = row[4]
            star = row[5]
            if user_id == 'user_id':
                continue
            history_u_lists[user_id] = business_id
            history_ur_lists[user_id] = star
            history_v_lists[business_id] = user_id
            history_vr_lists[business_id] = star
            train_u.append(user_id)
            train_v.append(business_id)
            train_r.append(star)
            # if i == 10000:
            #     break
            # break


    with open('history_u.pickle', 'wb') as f:
        a = [history_u_lists, history_ur_lists]
        pickle.dump(a, f)
    with open('history_v.pickle', 'wb') as f:
        a = [history_v_lists, history_vr_lists]
        pickle.dump(a, f)
    with open('train.pickle', 'wb') as f:
        a = [train_u, train_v, train_r]
        pickle.dump(a, f)

    # with open('data_set.pickle', 'rb') as f:
    #     reader = pickle.load(f)
    #     for row in reader:
    #         print(type(row))
    #         print(row)

if __name__ == '__main__':
    # t1 = time.clock()
    # user_graph, user_dict = create_user_graph()
    # t2 = time.clock()
    # user_item_graph = create_user_item_graph()
    # t3 = time.clock()
    # print(t2-t1)
    # print(t3-t2)
    # test_pickle()
    create_pickle()
    # with open('social_list.pickle', 'r') as f:
    #     reader = csv.reader(f)
    #     for row in reader:
    #         print(row)