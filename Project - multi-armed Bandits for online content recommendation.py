#Project - multi-armed Bandits for online headline news recommendation

#!/usr/bin/env python
# coding: utf-8

# In[114]:


import numpy as np
import pandas as pd
from numpy.random import binomial, randint, beta
from statistics import mean, median,variance,stdev
import random
import sys
import time
import math
import csv
from sklearn.cluster import KMeans


# In[115]:


# Info
# 1: 4681992
# 2: 3679695
# 3: 3966363
# 4: 5432561
# 5: 5377224
# 6: 5367570
# 7: 5203040
# 8: 4714618
# 9: 3618698
# 10: 3770122


# In[116]:


args = sys.argv
#args = {}


# In[117]:


start_time = time.time()


# In[118]:


# Parameter Settings
## Bandit parameters
if len(args) > 1:
    para_mode = int(args[1])
    if para_mode == 1:
        epsilon = float(args[2])
    elif para_mode == 2:
        alpha = float(args[2])
    elif para_mode == 3:
        b_ts = float(args[2])
    elif para_mode == 4:
        v_cb = float(args[2])

    day_id = int(args[3])
    num_c = int(args[4]) # Number of Clustering

else:
    para_mode = 3 # 0: random, 1: e-greedy, 2: UCB, 3: Thompson Sampling, 4: Thompson Sampling (Contextual Bandit)
    if para_mode == 1:
        epsilon = 0.4
    elif para_mode == 2:
        alpha = 1.0 # 3.0
    elif para_mode == 3:
        b_ts = 1.0 # 3.0
    elif para_mode == 4:
        v_cb = 0.05

    day_id = 2
    num_c = 1 # Number of Clustering

n_features = 12
#thompson_cb_r = 0.001
#thompson_cb_delta = 0.05
#thompson_cb_epsilon = 1.0 / float(math.log(n_features))

## Data parameters (Total: 4681992 for Day1, 15000 * 1000)
chunk_size = 30000 # 30000
chunk_loop_num = 200 # 200

num = chunk_size * chunk_loop_num


# In[119]:


tmp_arms = {}
total_reward = 0
total_count = 0


# In[120]:


## Data Settings
col_names = []
data_dtype = {}
col_names.append("unixtime")
col_names.append("news_id")
col_names.append("clicked")
col_names.append("user")
for i in range(2,7):
    col_names.append("user_feature_" + str(i))
    data_dtype["user_feature_" + str(i)] = object

col_names.append("user_feature_" + str(1))
data_dtype["user_feature_" + str(1)] = object

for j in range(20):
    col_names.append("news_" + str(j))
    data_dtype["news_" + str(j)] = object
    for i in range(2, 7):
        col_names.append("news_" + str(j) + "_feature_" + str(i))
        data_dtype["news_" + str(j) + "_feature_" + str(i)] = object
    col_names.append("news_" + str(j) + "_feature_" + str(1))
    data_dtype["news_" + str(j) + "_feature_" + str(1)] = object

for tmp in range(30):
    col_names.append("tmp" + str(tmp))
    data_dtype["tmp" + str(tmp)] = object

# In[121]:


fname_list = {}
fname_list[1] = 'ydata-fp-td-clicks-v1_0.20090501.gz'
fname_list[2] = 'ydata-fp-td-clicks-v1_0.20090502.gz'
fname_list[3] = 'ydata-fp-td-clicks-v1_0.20090503.gz'
fname_list[4] = 'ydata-fp-td-clicks-v1_0.20090504.gz'
fname_list[5] = 'ydata-fp-td-clicks-v1_0.20090505.gz'
fname_list[6] = 'ydata-fp-td-clicks-v1_0.20090506.gz'
fname_list[7] = 'ydata-fp-td-clicks-v1_0.20090507.gz'
fname_list[8] = 'ydata-fp-td-clicks-v1_0.20090508.gz'
fname_list[9] = 'ydata-fp-td-clicks-v1_0.20090509.gz'
fname_list[10] = 'ydata-fp-td-clicks-v1_0.20090510.gz'


# In[122]:


# Center of each cluster
cluster_g = {}
if num_c == 1:
    cluster_g[0] = pd.Series([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], index=['user_feature_2', 'user_feature_3', 'user_feature_4', 'user_feature_5', 'user_feature_6', 'user_feature_1'])
elif num_c == 2:
    cluster_g[0] = pd.Series([0.243250, 0.137371, 0.277737, 0.328405, 0.013237, 1.000000], index=['user_feature_2', 'user_feature_3', 'user_feature_4', 'user_feature_5', 'user_feature_6', 'user_feature_1'])
    cluster_g[1] = pd.Series([0.018438, 0.007748, 0.013008, 0.020511, 0.940295, 1.000000], index=['user_feature_2', 'user_feature_3', 'user_feature_4', 'user_feature_5', 'user_feature_6', 'user_feature_1'])
elif num_c == 3:
    cluster_g[0] = pd.Series([0.297865, 0.188523, 0.089325, 0.410124, 0.014163, 1.000000], index=['user_feature_2', 'user_feature_3', 'user_feature_4', 'user_feature_5', 'user_feature_6', 'user_feature_1'])
    cluster_g[1] = pd.Series([0.018477, 0.007567, 0.013011, 0.020419, 0.940526, 1.000000], index=['user_feature_2', 'user_feature_3', 'user_feature_4', 'user_feature_5', 'user_feature_6', 'user_feature_1'])
    cluster_g[2] = pd.Series([0.097470, 0.001237, 0.779543, 0.110533, 0.011217, 1.000000], index=['user_feature_2', 'user_feature_3', 'user_feature_4', 'user_feature_5', 'user_feature_6', 'user_feature_1'])
elif num_c == 4:
    cluster_g[0] = pd.Series([0.087859, 0.067672, 0.117595, 0.714820, 0.012054, 1.000000], index=['user_feature_2', 'user_feature_3', 'user_feature_4', 'user_feature_5', 'user_feature_6', 'user_feature_1'])
    cluster_g[1] = pd.Series([0.018234, 0.007241, 0.013056, 0.020562, 0.940908, 1.000000], index=['user_feature_2', 'user_feature_3', 'user_feature_4', 'user_feature_5', 'user_feature_6', 'user_feature_1'])
    cluster_g[2] = pd.Series([0.435244, 0.266179, 0.074256, 0.208537, 0.015785, 1.000000], index=['user_feature_2', 'user_feature_3', 'user_feature_4', 'user_feature_5', 'user_feature_6', 'user_feature_1'])
    cluster_g[3] = pd.Series([0.094922, 0.001260, 0.783986, 0.108470, 0.011362, 1.000000], index=['user_feature_2', 'user_feature_3', 'user_feature_4', 'user_feature_5', 'user_feature_6', 'user_feature_1'])


# In[123]:


#index = 0
#for r in reader:
#    if index == 300:
#        break
#    print(type(r), r.shape)
#    num_of_records += chunk_size
#
#    index += 1


# In[124]:


#df = reader.get_chunk(chunk_size)
#df.head().to_csv('20191121.csv')


# In[125]:


fname = fname_list[day_id]
reader = pd.read_csv(fname, sep = " ", chunksize=chunk_size, header = None, error_bad_lines=False, names=col_names, dtype = data_dtype)

chunk_loop = 0
loop_flag = 1
while chunk_loop < chunk_loop_num and loop_flag == 1:
    # Data Reading
    df = reader.get_chunk(chunk_size)

    if len(df) < chunk_size:
        loop_flag = 0

    for row in df.itertuples():
        # Data preparation
        try:
            flag_data = 0
            user_feature = {}
            arm_feature = {}
            for j in range(len(row)):
                value = str(row[j])
                if ('|' in value) and not '|user' in value:
                    flag_data += 1
                    tmp_arm_id = int(value.replace('|', ''))
                    arm_feature[tmp_arm_id] = {}

                if ':' in value:
                    if flag_data == 0:
                        user_feature[int(value.split(':')[0])] = float(value.split(':')[1])
                    else:
                        arm_feature[tmp_arm_id][int(value.split(':')[0])] = float(value.split(':')[1])
        except:
            continue

        if num_c > 1:
            diff_dict = {}
            for j in range(num_c):
                diff_dict[j] = 0.0
                for jj in range(1, 7):
                    diff_dict[j] += (user_feature[jj] - cluster_g[j]['user_feature_' + str(jj)])**2

            cluster_id = min(diff_dict, key=diff_dict.get)
        else:
            cluster_id = 0

        # candidate_dict: information about the candidate articles for each record
        candidate_dict = {}
        for tmp_arm_id in arm_feature:
            if len(arm_feature[tmp_arm_id]) == 6:
                candidate_dict[tmp_arm_id] = 0.0

        # if a new article is observed, a new key and value is added to "tmp_arms".
        for j in candidate_dict:
            if not j in tmp_arms:
                tmp_arms[j] = {}
                for jj in range(num_c):
                    tmp_arms[j][jj] = {}
                    tmp_arms[j][jj]["success"] = 0
                    tmp_arms[j][jj]["fail"] = 0
                    tmp_arms[j][jj]["success_rate"] = 0.0
                    tmp_arms[j][jj]["life"] = 0
                    tmp_arms[j][jj]["B"] = {}
                    tmp_arms[j][jj]["mu"] = {}
                    tmp_arms[j][jj]["f"] = {}

        for j in candidate_dict:
            tmp_arms[j][cluster_id]["life"] += 1

        if para_mode == 0:
            selected_arm, selected_arm_success_rate = random.choice(list(candidate_dict.items()))
        elif para_mode == 1:
            if binomial(n=1, p=epsilon) == 1:
                # Exploration (selection an article randomly)
                selected_arm, selected_arm_success_rate = random.choice(list(candidate_dict.items()))

            else:
                # Exploitation (selecting the best article)
                for j in candidate_dict:
                    tmp_cnt = tmp_arms[j][cluster_id]["success"] + tmp_arms[j][cluster_id]["fail"]
                    if tmp_cnt > 0:
                        candidate_dict[j] = float(tmp_arms[j][cluster_id]["success"])/float(tmp_cnt)
                    else:
                        candidate_dict[j] = 0.0

                max_k_list = [k[0] for k in candidate_dict.items() if k[1] == max(candidate_dict.values())]
                selected_arm = random.choice(max_k_list)
        elif para_mode == 2:
            for j in candidate_dict:
                tmp_cnt = tmp_arms[j][cluster_id]["success"] + tmp_arms[j][cluster_id]["fail"]
                ucb = math.sqrt(2*math.log(total_count+1) / float(tmp_cnt+1)) * alpha
                if tmp_cnt > 0:
                    candidate_dict[j] = float(tmp_arms[j][cluster_id]["success"]) / float(tmp_cnt) + ucb
                else:
                    candidate_dict[j] = ucb

                #print (j, total_count, tmp_cnt, ucb, len(candidate_dict))

            max_k_list = [k[0] for k in candidate_dict.items() if k[1] == max(candidate_dict.values())]
            selected_arm = random.choice(max_k_list)
            #print (max_k_list, candidate_dict)
            #print ("selected", selected_arm)

        elif para_mode == 3:
            for j in candidate_dict:
                tmp_a = (tmp_arms[j][cluster_id]["success"]+1)*b_ts
                tmp_b = (tmp_arms[j][cluster_id]["fail"]+1)*b_ts
                candidate_dict[j] = beta(a=tmp_a, b=tmp_b)

            max_k_list = [k[0] for k in candidate_dict.items() if k[1] == max(candidate_dict.values())]
            selected_arm = random.choice(max_k_list)

        elif para_mode == 4:
            for j in candidate_dict:
                tmp_cnt = tmp_arms[j][cluster_id]["success"] + tmp_arms[j][cluster_id]["fail"]
                if tmp_cnt == 0:
                    tmp_arms[j][cluster_id]["B"] = np.identity(n_features)
                    tmp_arms[j][cluster_id]["mu"] = np.zeros(n_features).reshape(n_features, 1)
                    tmp_arms[j][cluster_id]["f"] = np.zeros(n_features).reshape(n_features, 1)

                tmp_list = []
                for kk in range(6):
                    tmp_list.append(user_feature[kk+1])
                for kk in range(6):
                    tmp_list.append(arm_feature[j][kk+1])

                user = np.array(tmp_list)
                b = np.array(user).reshape(n_features, 1)
                mean = tmp_arms[j][cluster_id]["mu"].T.tolist()[0]
                #v = thompson_cb_r * math.sqrt((1/float(thompson_cb_epsilon))*n_features*math.log(1/float(thompson_cb_delta)))
                v = v_cb
                cov = (v**2)*np.linalg.inv(tmp_arms[j][cluster_id]["B"])

                sample_mu = np.random.multivariate_normal(mean, cov, 1).T
                candidate_dict[j] = np.dot(b.T, sample_mu)

            max_k_list = [k[0] for k in candidate_dict.items() if k[1] == max(candidate_dict.values())]
            selected_arm = random.choice(max_k_list)

            #print (max_k_list, candidate_dict)
            #print ("selected", selected_arm)

        else:
            sys.exit("para_mode is wrong!")

        # checking the results
        # if "row.news_id" is equal to "selected_arm", we evaluate the performance of the arm and then set "flag" to be 1 to select a new arm in the next record.
        # if "row.news_id" is NOT equal to "selected_arm", we don't evaluate the performance of the arm and then set "flag" to be 0 not to select a new arm in the next record.
        if row.news_id == selected_arm:
            if row.clicked == 1:
                tmp_arms[selected_arm][cluster_id]["success"] += 1
            else:
                tmp_arms[selected_arm][cluster_id]["fail"] += 1

            total_reward += row.clicked
            total_count += 1

            if para_mode == 4:
                tmp_list = []
                for kk in range(6):
                    tmp_list.append(user_feature[kk+1])
                for kk in range(6):
                    tmp_list.append(arm_feature[selected_arm][kk+1])

                user = np.array(tmp_list)
                b = np.array(user).reshape(n_features, 1)
                tmp_arms[selected_arm][cluster_id]["B"] += np.dot(b, b.T)
                tmp_arms[selected_arm][cluster_id]["f"] += row.clicked * b
                tmp_arms[selected_arm][cluster_id]["mu"] = np.dot(np.linalg.inv(tmp_arms[selected_arm][cluster_id]["B"]), tmp_arms[selected_arm][cluster_id]["f"])

    print ("loop num: " + str(chunk_loop) + ", time: " + str(time.time() - start_time))
    chunk_loop += 1


# In[126]:


if para_mode == 0:
    print ("Algorithm: random")
elif para_mode == 1:
    print ("Algorithm: e-greedy, epsilon = " + str(epsilon))
elif para_mode == 2:
    print ("Algorithm: UCB, alpha = " + str(alpha))
elif para_mode == 3:
    print ("Algorithm: Thompson Sampling, b_ts = " + str(b_ts))
elif para_mode == 4:
    print ("Algorithm: Contextual Bandit, Thompson Sampling, v = " + str(v_cb))

print ("Total count: " + str(total_count))
print ("Total reward: " + str(total_reward))
print ("TOtal click through rate: " + str(float(total_reward)/float(total_count)))


# In[127]:


if para_mode == 1:
    output_file_name = 'Day' + str(day_id) + '_mode' + str(para_mode) + '_e' + str(epsilon) + '_num' + str(num) + '_file' + str(len(fname_list)) + '_cluster' + str(num_c) + '.csv'
elif para_mode == 2:
    output_file_name = 'Day' + str(day_id) + '_mode' + str(para_mode) + '_a' + str(alpha) + '_num' + str(num) + '_file' + str(len(fname_list)) + '_cluster' + str(num_c) + '.csv'
elif para_mode == 3:
    output_file_name = 'Day' + str(day_id) + '_mode' + str(para_mode) + '_b' + str(b_ts) + '_num' + str(num) + '_file' + str(len(fname_list)) + '_cluster' + str(num_c) + '.csv'
elif para_mode == 4:
    output_file_name = 'Day' + str(day_id) + '_mode' + str(para_mode) + '_v' + str(v_cb) + '_num' + str(num) + '_file' + str(len(fname_list)) + '_cluster' + str(num_c) + '.csv'
else:
    output_file_name = 'Day' + str(day_id) + '_mode' + str(para_mode) + '_num' + str(num) + '_file' + str(len(fname_list)) + '_cluster' + str(num_c) + '.csv'

with open(output_file_name, 'a') as f:
    writer = csv.writer(f)
    writer.writerow(['Day ID: ' + str(day_id)])
    writer.writerow(['Number of clusters: ' + str(num_c)])
    if para_mode == 0:
        writer.writerow(['Algorithm: random'])
    elif para_mode == 1:
        writer.writerow(['Algorithm: e-greedy' + ', e: ' + str(epsilon)])
    elif para_mode == 2:
        writer.writerow(['Algorithm: UCB' + ', a: ' + str(alpha)])
    elif para_mode == 3:
        writer.writerow(['Algorithm: Thompson Sampling' + ', b: ' + str(b_ts)])
    elif para_mode == 4:
        writer.writerow(['Algorithm: Thompson Sampling (Contextual)' + ', v: ' + str(v_cb)])

    writer.writerow([])

    writer.writerow(['Total count: ', total_count])
    writer.writerow(['Total reward: ', total_reward])
    writer.writerow(['Total CTR: ', float(total_reward)/float(total_count)])
    writer.writerow([])

    for jj in range(num_c):
        writer.writerow(['Number of clusters: ' + str(jj)])
        writer.writerow(['news id', 'success', 'fail', 'rate', 'life'])
        for arm in tmp_arms:
            if (tmp_arms[arm][jj]["success"] + tmp_arms[arm][jj]["fail"]) > 0:
                tmp_rate = float(tmp_arms[arm][jj]["success"])/float(tmp_arms[arm][jj]["fail"])
            else:
                tmp_rate = 0.0

            writer.writerow([arm, tmp_arms[arm][jj]["success"], tmp_arms[arm][jj]["fail"], tmp_rate, tmp_arms[arm][jj]["life"]])

        writer.writerow([])


# In[128]:


elapsed_time = time.time() - start_time
print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")


# In[ ]:





# In[ ]:
