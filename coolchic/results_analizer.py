from lossless.util.config import TEST_WORKDIR, args
import os
import pandas as pd

results = []
for file in os.listdir(TEST_WORKDIR):
    if file.endswith("_rates.txt"):
        with open(os.path.join(TEST_WORKDIR, file), "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    results.append(line)


def result_line_to_dict(line:str) -> dict[str, float]:
    line = line.strip("{} \n")
    line = line.replace(", '", "; '")
    line = line.replace("'", "")
    parts = line.split("; ")
    result = {}
    for part in parts:
        key, value = part.split(": ")
        if value.startswith("tensor"):
            value = value[value.index("(")+1:value.index(",")]
        result[key] = float(value)
    return result

results_dicts = [result_line_to_dict(line) for line in results]
print(results_dicts)
df = pd.DataFrame(results_dicts)
print(df)
# print average of each column
print("Averages:")
print(df.mean())

df.to_csv(os.path.join(TEST_WORKDIR, "results_summary.csv"), index=False)

fnlic_results_path = "../../FNLIC/test/kodak_model_check.csv"
if os.path.exists(fnlic_results_path):
    fnlic_df = pd.read_csv(fnlic_results_path)
else:
    raise FileNotFoundError(f"FNLIC results file not found at {fnlic_results_path}")

# drop Image column
fnlic_df = fnlic_df.drop(columns=["Image", "Is_Same", "Latent_0_BPD", "Latent_1_BPD", "Latent_2_BPD", "Latent_3_BPD", "Img_0_BPD", "Img_1_BPD", "Img_2_BPD", "Img_3_BPD", "Img_4_BPD", "Img_5_BPD", "Img_6_BPD", "Img_7_BPD", "Img_8_BPD", "Img_9_BPD", "Img_10_BPD", "Img_11_BPD"])
# drop rows with index not in df
fnlic_df = fnlic_df[fnlic_df.index.isin(df.index)]
# rename column names Real_BPD  NN_BPD  Latent_BPD  Img_BPD to total_bpd network_bpd latent_bpd image_bpd
fnlic_df = fnlic_df.rename(columns={"Real_BPD": "total_bpd", "NN_BPD": "network_bpd", "Latent_BPD": "latent_bpd", "Img_BPD": "image_bpd"}) 
print("FNLIC results:")
print(fnlic_df)
print("FNLIC Averages:")
print(fnlic_df.mean())

difference_df = df - fnlic_df
print("Difference between Cool-Chic and FNLIC:")
print(difference_df)
print("Difference Averages:")
print(difference_df.mean())