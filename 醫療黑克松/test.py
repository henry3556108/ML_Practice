import os
import pandas as pd

csv_files = list(map(lambda x: os.path.join(
    "result", x), os.listdir("result")))
print(csv_files)
total = pd.DataFrame()

for csv_path in csv_files:
    df = pd.read_csv(csv_path, index_col = 0)
    total = pd.concat([total, df], axis=0)

total.to_csv("final_result.csv", index=False)

# import requests
# import json

# a = requests.request(method = "get" , url = "https://api.fda.gov/drug/ndc.json?search=generic_name:'methylprednisolone'&limit=100&skip=100")
# with open("methylprednisolone2.json", 'w') as f:
#     f.write(a.text)

# def parser(json):
#     ndc_codes = []
#     ndc_codes.append(int(json["product_ndc"].replace("-","")))
#     for sub in json["packaging"]:
#         ndc_codes.append(int(sub["package_ndc"].replace("-","")))
#         # print(str(ndc_codes))
#     with open("ndc.txt", "a+") as f:
#         # f.write(str(ndc_codes))
#         for ndc in ndc_codes:
#             f.write(str(ndc)+",")
#         f.write("\n")


# with open('methylprednisolone2.json', 'r') as f:
#     dic = json.load(fp=f)
#     # print(dic)
#     for pack in dic["results"]:
#         parser(pack)
# import os
# import pandas as pd
# dirs = os.listdir("test")
# result = pd.DataFrame()
# for f in dirs:
#     df = pd.read_csv("")


