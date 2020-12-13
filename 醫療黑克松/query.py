import pandas as pd
import numpy as np
# s = ""
# df = pd.read_csv("prescriptions.csv")
# ndcs = pd.read_csv("prednisone_ndc.csv",index_col=0)["NDC_code"]
# result = pd.DataFrame()
# for ndc in np.squeeze(ndcs.values):
#     df["NDC"].fillna(-1)
#     try:
#         assert(float(ndc))
#         temp = df[df["NDC"]==float(ndc)]
#         result = pd.concat([result, temp], axis=0)  
#         print(result, result.shape)
#     except:
#         pass
# result.to_csv("哲嘉_result_test.csv", index=False)

prescriptions = pd.read_csv("prescriptions.csv")
# print(prescriptions.head(10))
patient = pd.read_csv("patient_final.csv")

targetPatientPrescription_v2 = prescriptions[prescriptions["SUBJECT_ID"] in patient['SUBJECT_ID'].values]

print(targetPatientPrescription_v2.head(10))



# print(ndc)
# with open("test/ndc.txt", "r") as f:
#     s = s.join(f.readlines()).replace("\n","")
#     s = s.split(",")
#     result = pd.DataFrame()
#     df["NDC"].fillna(-1)
#     for sub in s:
#         try:
#             assert(float(sub))
#             temp = df[df["NDC"]==float(sub)]
#             result = pd.concat([result, temp], axis=0)  
#             print(result, result.shape)
#         except:
#             pass
#     result.to_csv("result.csv", index=False)


