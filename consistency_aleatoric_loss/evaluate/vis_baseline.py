from tabulate import tabulate
import json
table = []
# json_file_path = './results.json'
json_file_path = './baseline_results.json'

with open(json_file_path) as f:
    current_result_dic = json.load(f)

# headers = ["dataset","Smeasure","meanFm","meanEm","MAE","time"]
# ####################
# for key in current_result_dic.keys():
#     print(key) #data path
#     for keys in current_result_dic[key].keys():
#         values_list = list(current_result_dic[key][keys].values())
#         table.append([keys]+values_list)
#
# my_table = tabulate(table, headers, tablefmt="fancy_grid")
# print(my_table)

# print('#############################################')
###################
new_table = []
table_list = [[],[],[],[]]
new_header = ["method","Smeasure","meanFm","meanEm","MAE","time"]

for i in current_result_dic:
    for j_index,j in enumerate(current_result_dic[i]):
        table_list[j_index].append([i]+list(current_result_dic[i][j].values()))

for index,new_table in enumerate(table_list):
    print(['CAMO', 'CHAMELEON', 'COD10K', 'NC4K'][index])
    another_table = tabulate(new_table, new_header, tablefmt="fancy_grid")
    print(another_table)