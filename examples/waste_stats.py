from sarc.client.series import load_job_series

filename = "mila_job_series4.pkl"
df = load_job_series(filename)

# Group jobs by user
grouped_by_user = df.groupby(["user"])

# Compute the total amount of compute time used, wasted and overbilled by user.
df["used"] = df["value"] / 100.0 * df["duration"]
df["wasted"] = (1 - df["value"] / 100.0) * df["duration"]
df["overbilled"] = (df["allocated"] - df["requested"]) * df["duration"]

waste_by_user = df.groupby(["user"])["wasted"].sum()
usage_by_user = df.groupby("user")["used"].sum()
overbilling_by_user = df.groupby("user")["overbilled"].sum()

# Compute the ratios of wasted time to used time. Close to 0 is good, between 0.5 and 1 is concerning, above 1 is bad.
ratios = waste_by_user / usage_by_user
ratios = ratios.reset_index().rename(columns={0: "ratio"})
ratios["wasted"] = waste_by_user.values
ratios["overbilled"] = overbilling_by_user.values
ratios["used"] = usage_by_user.values
ratios["total_wasted"] = ratios["wasted"] + ratios["overbilled"]
ratios["total_ratio"] = (ratios["wasted"] + ratios["overbilled"]) / ratios["used"]

# Print from worst offender to best usage.
print(ratios.reset_index().sort_values(ascending=False, by="total_ratio"))
