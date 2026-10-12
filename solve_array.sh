#!/bin/bash
#SBATCH --job-name=gurobi_solve
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6                # ggf. an Threads-Setting in deinen Methoden anpassen
#SBATCH --mem=66G                          # TODO: an Modellgroesse anpassen
#SBATCH --time=02:40:00                   # TODO: deutlich > internes --time-limit unten
#SBATCH --partition=cpu
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=julius.hoffmann@kit.edu                     # TODO: deine Mail-Adresse
# --array wird NICHT hier gesetzt, sondern von submit_all.sh beim
# sbatch-Aufruf dynamisch mitgegeben (haengt von Instanzen x Methoden ab).

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")}"
source config.sh

module load "$PYTHON_MODULE"
module load "$GUROBI_MODULE"
source "$VENV_DIR/bin/activate"

# Jeder Task bekommt GENAU EINE (Instanz, Methode, nr_fac)-Kombination, aufgeloest
# aus der Slurm-Array-Task-ID (siehe grid.py -> resolve_grid_task).
# Das ist die eigentliche "Stueckelung": statt eines Jobs, der alle
# Methoden fuer alle Instanzen nacheinander abarbeitet (Tage!), laufen
# hier viele einzelne, kurze Kombinationen parallel.

# Zeitbudget pro Task (muss in die Slurm-Walltime --time=02:00:00 passen):
#   Warten auf ein freies Gurobi-Token   max.  900 s (--license-wait)
#   + Gurobi-Rechenzeit                       5400 s (--time-limit)
#   = 6300 s (1h45)  ->  15 Minuten Reserve fuer Einlesen/Precalculation.
# Die Wartezeit zaehlt zur Walltime, weil der Task waehrenddessen bereits
# laeuft. Aendert ihr eine der Zahlen, --time oben mit anpassen.
# Vor dem Rechnen wird ein Token reserviert (siehe gurobi_license.py); ist
# keines frei, wird alle ~60 s erneut versucht. Klappt es bis zum Ablauf
# nicht, steht status=NO_LICENSE in der Ergebnisdatei.
python main.py --task-id "$SLURM_ARRAY_TASK_ID" --instance-dir instances \
    --time-limit 7200 --license-wait 900 --license-poll 60
