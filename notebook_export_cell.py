# ---------- Extra cell: export small files for the Streamlit dashboard ----------
# Paste this as a NEW cell at the END of the notebook (after daily_master_clean.csv is saved).
# It needs the variables already created above: daily, quality_report, hourly_steps,
# hourly_cal, heart_rate, OUT_DIR.

# 1) Per-user, per-hour sums and counts (lets the dashboard pool exactly and filter by user)
hs_u = (hourly_steps.assign(Hour=hourly_steps["ActivityHour"].dt.hour)
        .groupby(["Id", "Hour"])["StepTotal"].agg(steps_sum="sum", steps_n="count").reset_index())

hc_u = (hourly_cal.assign(Hour=hourly_cal["ActivityHour"].dt.hour)
        .groupby(["Id", "Hour"])["Calories"].agg(cal_sum="sum", cal_n="count").reset_index())

hr_u = (heart_rate.assign(Hour=heart_rate["Time"].dt.hour, sq=heart_rate["Value"].astype("float64") ** 2)
        .groupby(["Id", "Hour"]).agg(hr_sum=("Value", "sum"), hr_sumsq=("sq", "sum"), hr_n=("Value", "count"))
        .reset_index())

hourly_profile = (hs_u.merge(hc_u, on=["Id", "Hour"], how="outer")
                      .merge(hr_u, on=["Id", "Hour"], how="outer")
                      .fillna(0))

# 2) Write everything (daily file and quality report are already saved in section 5.3)
daily.to_csv(OUT_DIR + "daily_master_clean.csv", index=False)
quality_report.to_csv(OUT_DIR + "data_quality_report.csv", index=False)
hourly_profile.to_csv(OUT_DIR + "hourly_profile.csv", index=False)

print("Exported:", hourly_profile.shape, "-> hourly_profile.csv")

# 3) Optional: download the three files from Colab
# from google.colab import files
# for f in ["daily_master_clean.csv", "data_quality_report.csv", "hourly_profile.csv"]:
#     files.download(OUT_DIR + f)
