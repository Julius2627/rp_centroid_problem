"""
grid.py -- gemeinsame Hilfsfunktionen fuer die Aufteilung der Arbeit in
(Instanz, Methode)-Kombinationen ("chunking"). Wird sowohl von main.py
(zum Ausfuehren einer einzelnen Kombination) als auch von
collect_results.py (zum Erkennen fehlender Kombinationen) genutzt --
so verwenden beide garantiert dieselbe Reihenfolge und Zaehlweise.

Warum ueberhaupt aufteilen: main.py fuehrt selbst KEINE Schleife ueber
mehrere Instanzen/Methoden aus. Jeder Aufruf bearbeitet genau eine
Kombination. Die eigentliche Parallelisierung passiert auf Slurm-Ebene
durch den Array-Job (solve_array.sh) -- dadurch laufen z.B. 200
Instanz/Methode-Kombinationen gleichzeitig auf 200 Tasks, statt
nacheinander auf einem einzigen Knoten (was Tage dauern wuerde).
"""
import pkgutil
from pathlib import Path

import methods

NR_FACILITIES = [10,15,20]

def discover_instances(instance_dir):
    """Alle Instanz-CSVs, sortiert (stabile, reproduzierbare Reihenfolge)."""
    return sorted(Path(instance_dir).glob("*.csv"))


def discover_methods():
    """Alle Methoden-Module in methods/, sortiert. Dateiname (ohne .py)
    ist der Methodenname, z.B. methods/exact_mip.py -> "exact_mip"."""
    return sorted(
        name
        for _, name, is_pkg in pkgutil.iter_modules(methods.__path__)
        if not is_pkg and not name.startswith("_")
    )


def grid_size(instance_dir) -> int:
    """Gesamtzahl (Instanzen x Methoden x Fac_Values) -- das ist die benoetigte
    Slurm-Array-Groesse."""
    return len(discover_instances(instance_dir)) * len(discover_methods()) * len(NR_FACILITIES)


def resolve_grid_task(task_id: int, instance_dir):
    """0-basierte Task-ID (== $SLURM_ARRAY_TASK_ID) -> (instance_path, model_name).

    Reihenfolge: task_id = instance_idx * n_models * len(NR_FACILITIES) + model_idx * len(NR_FACILITIES) + nr_fac_idx
    (divmod mit n_models als Divisor)."""
    instances = discover_instances(instance_dir)
    models = discover_methods()

    if not instances:
        raise ValueError(f"Keine Instanzen in {instance_dir} gefunden.")
    if not models:
        raise ValueError("Keine Methoden in methods/ gefunden.")

    n_models = len(models)
    total = len(instances) * n_models * len(NR_FACILITIES)
    if not (0 <= task_id < total):
        raise ValueError(
            f"task-id {task_id} liegt ausserhalb des gueltigen Bereichs (0..{total - 1})."
        )

    instance_idx, model_and_len_ps_idx = divmod(task_id, n_models*len(NR_FACILITIES))
    model_idx, nr_fac_idx = divmod(model_and_len_ps_idx, len(NR_FACILITIES))
    return instances[instance_idx], models[model_idx], NR_FACILITIES[nr_fac_idx]


def result_stem(instance_path, model_name: str, nr_fac: int) -> str:
    """Eindeutiger Basisname (ohne .csv) fuer eine (Instanz, Methode)-Kombination."""
    return f"{Path(instance_path).stem}__{model_name}__p{nr_fac}"

