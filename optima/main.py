import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyomo.environ import *
from pyomo.opt import SolverStatus, TerminationCondition

# === Load Excel File ===
BASE_DIR = os.path.dirname(__file__)
FILE_NAME = "example_file.xlsx"
FILE_PATH = os.path.join(BASE_DIR, FILE_NAME)
RESULT_DIR = os.path.join(BASE_DIR, "Result")

# Create result folder
os.makedirs(RESULT_DIR, exist_ok=True)

xls = pd.ExcelFile(FILE_PATH)

demand_df = xls.parse("Demand")
process_df = xls.parse("Process")
process_commodity_df = xls.parse("Process-Commodity")
ext_import_df = xls.parse("Ext-Import")
ext_export_df = xls.parse("Ext-Export") if "Ext-Export" in xls.sheet_names else None


time_steps = demand_df['Time'].tolist()
commodities = ['elec', 'heat', 'gas']
processes = process_df['Process'].tolist()

# === Pyomo Model ===
model = ConcreteModel()
model.T = RangeSet(0, len(time_steps) - 1)
model.P = Set(initialize=processes)
model.C = Set(initialize=commodities)

# === Parameters ===
demand = {(t, c): 0 for t in model.T for c in commodities}
for t in model.T:
    demand[(t, 'elec')] = demand_df.loc[t, 'elec']
    demand[(t, 'heat')] = demand_df.loc[t, 'heat']

capex = process_df.set_index('Process')['cost-inv'].to_dict()
var_cost = process_df.set_index('Process')['cost-var'].to_dict()

proc_comm = process_commodity_df.copy()
proc_ratio = {(row['Process'], row['Commodity'], row['Direction']): row['ratio']
              for _, row in proc_comm.iterrows()}

import_price = {(t, c): ext_import_df.loc[t, c] for t in model.T for c in ['elec', 'gas']}

if ext_export_df is not None and not ext_export_df.empty:
    export_price = {(t, c): ext_export_df.loc[t, c] if c in ext_export_df.columns else 0
                    for t in model.T for c in commodities}
else:
    export_price = {(t, c): 0 for t in model.T for c in commodities}

# === Variables (bounded) ===
model.Cap = Var(model.P, within=NonNegativeReals, bounds=(0, 1e5))
model.Gen = Var(model.P, model.T, within=NonNegativeReals)
model.Imp = Var(model.T, model.C, within=NonNegativeReals, bounds=(0, 1e6))
model.Exp = Var(model.T, model.C, within=NonNegativeReals, bounds=(0, 1e6))

# === Objective ===
def total_cost_rule(m):
    investment = sum(capex[p] * m.Cap[p] for p in m.P)
    variable = sum(var_cost[p] * m.Gen[p, t] for p in m.P for t in m.T)
    imports = sum(m.Imp[t, c] * import_price.get((t, c), 0) for t in m.T for c in ['elec', 'gas'])
    exports = sum(m.Exp[t, c] * export_price.get((t, c), 0) for t in m.T for c in m.C)
    return investment + variable + imports - exports
model.TotalCost = Objective(rule=total_cost_rule, sense=minimize)

# === Constraints ===
def demand_balance(m, t, c):
    prod = sum(m.Gen[p, t] * proc_ratio.get((p, c, 'Out'), 0) for p in m.P)
    cons = sum(m.Gen[p, t] * proc_ratio.get((p, c, 'In'), 0) for p in m.P)
    return prod + m.Imp[t, c] >= demand[t, c] + m.Exp[t, c] + cons
model.DemandBalance = Constraint(model.T, model.C, rule=demand_balance)

def gen_cap(m, p, t):
    return m.Gen[p, t] <= m.Cap[p]
model.GenCap = Constraint(model.P, model.T, rule=gen_cap)

def input_fuel_rule(m, t):
    used = sum(m.Gen[p, t] * proc_ratio.get((p, 'gas', 'In'), 0) for p in m.P)
    return m.Imp[t, 'gas'] + 1e-6 >= used
model.InputFuel = Constraint(model.T, rule=input_fuel_rule)

# === Solve ===
solver = SolverFactory('cbc')
result = solver.solve(model, tee=True)

# === Check Solver Status ===
if (result.solver.status != SolverStatus.ok) or (result.solver.termination_condition != TerminationCondition.optimal):
    print("Solver did not find an optimal solution.")
    print("Status:", result.solver.status)
    print("Termination:", result.solver.termination_condition)
else:
    print("\nSolver found optimal solution.")

# === Output Results ===
cap_result = {p: model.Cap[p].value for p in model.P if model.Cap[p].value is not None and model.Cap[p].value > 1e-3}
print(" Optimal Device Capacities:")
for p, val in cap_result.items():
    print(f"  {p}: {val:.2f} kW")

# === Time Series Results ===
gen_df = pd.DataFrame({p: [model.Gen[p, t].value for t in model.T] for p in model.P})
imp_df = pd.DataFrame({c: [model.Imp[t, c].value for t in model.T] for c in commodities})
exp_df = pd.DataFrame({c: [model.Exp[t, c].value for t in model.T] for c in commodities})

for df in [gen_df, imp_df, exp_df]:
    df['Time'] = time_steps

# === Save Plots ===
def save_plot(df, cols, title, ylabel, filename):
    plt.figure(figsize=(12, 6))
    for col in cols:
        plt.plot(df['Time'], df[col], label=col)
    plt.xlabel("Time")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, filename))
    plt.close()

save_plot(gen_df, model.P, "Energy Generation per Process", "kWh", "generation.png")
save_plot(imp_df, commodities, "Imported Commodities", "kWh", "imports.png")
if exp_df.drop(columns=["Time"]).sum().sum() > 0:
    save_plot(exp_df, commodities, "Exported Commodities", "kWh", "exports.png")


# === Save Capacities and Time Series Data ===
cap_df = pd.DataFrame(list(cap_result.items()), columns=["Process", "Installed_Capacity_kW"])

excel_path = os.path.join(RESULT_DIR, "techno_economic_results.xlsx")
with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
    cap_df.to_excel(writer, sheet_name="Capacities", index=False)
    gen_df.to_excel(writer, sheet_name="Generation", index=False)
    imp_df.to_excel(writer, sheet_name="Imports", index=False)
    exp_df.to_excel(writer, sheet_name="Exports", index=False)

print(f"All results saved to: {RESULT_DIR}")

