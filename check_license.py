#!/usr/bin/env python3
"""
check_license.py -- prueft, ob Gurobi eine echte (unbeschraenkte) Lizenz findet.

Ein leeres Modell laeuft auch mit der groessenbeschraenkten "Restricted
license" (Fallback, wenn KEINE gueltige Lizenz gefunden wird) problemlos
durch. Deshalb wird hier bewusst ein Modell mit 3000 Variablen geloest:
das schlaegt mit der Restricted-Lizenz fehl (GurobiError 10010), mit einer
echten Lizenz (auch ueber Token-Server) nicht.

Exit-Code 0 = Lizenz OK, 1 = keine gueltige Lizenz.

Aufruf (Umgebung vorher laden: Gurobi-Modul + venv):
    python check_license.py
"""
import sys

import gurobipy as gp
from gurobipy import GRB


def main() -> int:
    print("gurobipy Version:", gp.gurobi.version())

    m = gp.Model()
    m.Params.OutputFlag = 0
    x = m.addVars(3000, vtype=GRB.BINARY)
    m.setObjective(gp.quicksum(x[i] for i in range(3000)), GRB.MAXIMIZE)
    m.addConstr(gp.quicksum(x[i] for i in range(3000)) <= 10)

    try:
        m.optimize()
    except gp.GurobiError as e:
        print()
        print("FEHLER: keine gueltige (unbeschraenkte) Gurobi-Lizenz nutzbar.")
        print("Details:", e)
        print("Siehe README, Abschnitt 'Lizenz'.")
        return 1

    print("Lizenz OK (Modell mit 3000 Variablen geloest).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
