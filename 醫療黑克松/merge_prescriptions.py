import pandas as pd


patient = pd.read_csv("patient_final.csv", index_col=0)
prescriptionsResult = pd.read_csv("prescriptions_result.csv")
subjectId = patient['SUBJECT_ID']
# print(prescriptionsResult["SUBJECT_ID"])
patientPrescription = prescriptionsResult[prescriptionsResult["SUBJECT_ID"].apply(lambda x : x in subjectId.values)]
patientPrescription.to_csv("targetPatientPrescription.csv", index=False)