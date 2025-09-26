from lossless.util.config import TEST_WORKDIR, args
import os
import pandas as pd


def result_line_to_dict(line: str) -> dict[str, float]:
    line = line.strip("{} \n")
    line = line.replace(", '", "; '")
    line = line.replace("'", "")
    parts = line.split("; ")
    result = {}
    for part in parts:
        key, value = part.split(": ")
        if value.startswith("tensor"):
            value = value[value.index("(") + 1 : value.index(",")]
        result[key] = float(value)
    return result


# results = []
# for file in os.listdir(TEST_WORKDIR):
#     if file.endswith("_rates.txt"):
#         with open(os.path.join(TEST_WORKDIR, file), "r") as f:
#             for line in f:
#                 line = line.strip()
#                 if line:
#                     results.append(line)
# results_dicts = [result_line_to_dict(line) for line in results]
# print(results_dicts)

results_dicts = [
    {
        "total_bpd": 2.3738,
        "image_bpd": 2.1358,
        "latent_bpd": 0.2104,
        "network_bpd": 0.02758110873401165,
    },
    {
        "total_bpd": 2.7645,
        "image_bpd": 2.5091,
        "latent_bpd": 0.2282,
        "network_bpd": 0.02724117785692215,
    },
    {
        "total_bpd": 3.1561,
        "image_bpd": 2.7605,
        "latent_bpd": 0.3677,
        "network_bpd": 0.0278481375426054,
    },
    {
        "total_bpd": 2.7447,
        "image_bpd": 2.3860,
        "latent_bpd": 0.3319,
        "network_bpd": 0.02675204910337925,
    },
    {
        "total_bpd": 3.2413,
        "image_bpd": 2.9101,
        "latent_bpd": 0.3038,
        "network_bpd": 0.0273140799254179,
    },
    {
        "total_bpd": 2.9406,
        "image_bpd": 2.5641,
        "latent_bpd": 0.3464,
        "network_bpd": 0.03001912496984005,
    },
    {
        "total_bpd": 2.5479,
        "image_bpd": 2.3465,
        "latent_bpd": 0.1739,
        "network_bpd": 0.02747514471411705,
    },
    {
        "total_bpd": 3.3536,
        "image_bpd": 3.0837,
        "latent_bpd": 0.2447,
        "network_bpd": 0.02518039382994175,
    },
    {
        "total_bpd": 2.7926,
        "image_bpd": 2.3507,
        "latent_bpd": 0.4140,
        "network_bpd": 0.02788713201880455,
    },
    {
        "total_bpd": 2.8054,
        "image_bpd": 2.5620,
        "latent_bpd": 0.2167,
        "network_bpd": 0.0266859270632267,
    },
    {
        "total_bpd": 2.7788,
        "image_bpd": 2.4187,
        "latent_bpd": 0.3344,
        "network_bpd": 0.0256517194211483,
    },
    {
        "total_bpd": 2.6567,
        "image_bpd": 2.3103,
        "latent_bpd": 0.3160,
        "network_bpd": 0.03040567971765995,
    },
    {
        "total_bpd": 3.5251,
        "image_bpd": 3.1534,
        "latent_bpd": 0.3462,
        "network_bpd": 0.02553727850317955,
    },
    {
        "total_bpd": 3.0710,
        "image_bpd": 2.8864,
        "latent_bpd": 0.1596,
        "network_bpd": 0.02502950094640255,
    },
    {
        "total_bpd": 2.6036,
        "image_bpd": 2.2590,
        "latent_bpd": 0.3142,
        "network_bpd": 0.0303005650639534,
    },
    {
        "total_bpd": 2.7204,
        "image_bpd": 2.4720,
        "latent_bpd": 0.2215,
        "network_bpd": 0.02686055563390255,
    },
    {
        "total_bpd": 2.7743,
        "image_bpd": 2.4013,
        "latent_bpd": 0.3443,
        "network_bpd": 0.02875179797410965,
    },
    {
        "total_bpd": 3.3238,
        "image_bpd": 3.0646,
        "latent_bpd": 0.2344,
        "network_bpd": 0.02483283169567585,
    },
    {
        "total_bpd": 3.0564,
        "image_bpd": 2.6659,
        "latent_bpd": 0.3624,
        "network_bpd": 0.0280439592897892,
    },
    {
        "total_bpd": 2.1509,
        "image_bpd": 1.9619,
        "latent_bpd": 0.1621,
        "network_bpd": 0.02684953436255455,
    },
    {
        "total_bpd": 2.9772,
        "image_bpd": 2.6415,
        "latent_bpd": 0.3108,
        "network_bpd": 0.02483452670276165,
    },
    {
        "total_bpd": 3.1106,
        "image_bpd": 2.6442,
        "latent_bpd": 0.4391,
        "network_bpd": 0.0273403599858284,
    },
    {
        "total_bpd": 2.5819,
        "image_bpd": 2.4085,
        "latent_bpd": 0.1473,
        "network_bpd": 0.02605183981359005,
    },
    {
        "total_bpd": 2.9118,
        "image_bpd": 2.4409,
        "latent_bpd": 0.4425,
        "network_bpd": 0.028411865234375,
    },
]


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
    raise FileNotFoundError(
        f"FNLIC results file not found at {fnlic_results_path}"
    )

# drop Image column
fnlic_df = fnlic_df.drop(
    columns=[
        "Image",
        "Is_Same",
        "Latent_0_BPD",
        "Latent_1_BPD",
        "Latent_2_BPD",
        "Latent_3_BPD",
        "Img_0_BPD",
        "Img_1_BPD",
        "Img_2_BPD",
        "Img_3_BPD",
        "Img_4_BPD",
        "Img_5_BPD",
        "Img_6_BPD",
        "Img_7_BPD",
        "Img_8_BPD",
        "Img_9_BPD",
        "Img_10_BPD",
        "Img_11_BPD",
    ]
)
# drop rows with index not in df
fnlic_df = fnlic_df[fnlic_df.index.isin(df.index)]
# rename column names Real_BPD  NN_BPD  Latent_BPD  Img_BPD to total_bpd network_bpd latent_bpd image_bpd
fnlic_df = fnlic_df.rename(
    columns={
        "Real_BPD": "total_bpd",
        "NN_BPD": "network_bpd",
        "Latent_BPD": "latent_bpd",
        "Img_BPD": "image_bpd",
    }
)
print("FNLIC results:")
print(fnlic_df)
print("FNLIC Averages:")
print(fnlic_df.mean())

difference_df = df - fnlic_df
print("Difference between Cool-Chic and FNLIC:")
print(difference_df)
print("Difference Averages:")
print(difference_df.mean())
