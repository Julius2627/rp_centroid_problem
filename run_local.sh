#!/bin/bash
# run_local.sh -- alle (Instanz, Methode, p)-Kombinationen auf EINEM normalen
# Rechner (Linux) abarbeiten, mehrere gleichzeitig. Gegenstueck zu
# submit_all.sh + solve_array.sh + merge_results.sh auf dem Cluster.
#
# Jeder Durchlauf bekommt hart begrenzt:
#   * 7 Kerne      -> taskset (feste CPU-Zuordnung) + OMP/BLAS-Thread-Variablen
#   * 40 GB RAM    -> cgroup (systemd-run), sonst Fallback ulimit -v
# Die Grenzen gelten fuer den GANZEN Prozess, also Python/numpy UND Gurobi.
# Bei 24 Kernen laufen so bis zu 3 Durchlaeufe gleichzeitig (siehe unten).
#
# Aufruf (im Projektordner, dort wo main.py liegt):
#   ./run_local.sh -j 4               # 4 Durchlaeufe gleichzeitig (alle Tasks)
#   ./run_local.sh -j 3 0-399         # 3 gleichzeitig, nur Task-IDs 0 bis 399
#   ./run_local.sh -j 2 403 722 729   # 2 gleichzeitig, nur einzelne IDs (auch "403,722,729")
#   ./run_local.sh                    # ohne -j: Standard aus DEFAULT_JOBS (siehe Einstellungen)
#   nohup ./run_local.sh -j 4 > run_local.log 2>&1 &   # laeuft auch nach Logout weiter
#
# Optionen:
#   -j, --jobs N         wie viele Durchlaeufe GLEICHZEITIG laufen (Standard: DEFAULT_JOBS = 3)
#   -c, --cores N        Kerne pro Durchlauf                        (Standard: 7)
#   -m, --mem GB         RAM pro Durchlauf in GB (harte Grenze)     (Standard: 40)
#   -t, --time-limit S   Gurobi-Zeitlimit pro Durchlauf in Sekunden (Standard: 7200)
#   -h, --help           diese Hilfe
# Alles davon geht auch per Umgebungsvariable (JOBS=4 ./run_local.sh);
# die Option auf der Kommandozeile hat Vorrang.

set -o pipefail   # bewusst KEIN "set -e": ein fehlgeschlagener Task darf die anderen nicht stoppen

cd "$(dirname "${BASH_SOURCE[0]}")" || exit 1

usage() { sed -n '/^# Aufruf/,/^# die Option auf der Kommandozeile/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

# ---------------------------------------------------------------- Einstellungen
CORES_PER_JOB=${CORES_PER_JOB:-7}       # Kerne pro Durchlauf
MEM_PER_JOB_GB=${MEM_PER_JOB_GB:-40}    # RAM pro Durchlauf in GB (harte Grenze)
TIME_LIMIT=${TIME_LIMIT:-7200}          # Gurobi-Zeitlimit pro Durchlauf in Sekunden
INSTANCE_DIR=${INSTANCE_DIR:-instances}
PYTHON=${PYTHON:-python3}
VENV_DIR=${VENV_DIR:-.venv}             # wird aktiviert, falls vorhanden
# Die Lizenz-Wartelogik ist fuer den Token-Server des Clusters. Auf dem eigenen
# Rechner (lokale Lizenz) ist sie unnoetig -> standardmaessig aus.
SKIP_LICENSE_CHECK=${SKIP_LICENSE_CHECK:-1}
DEFAULT_JOBS=3                          # <<< HIER aendern: so viele Durchlaeufe laufen standardmaessig gleichzeitig
JOBS=${JOBS:-}                          # leer = DEFAULT_JOBS; per -j oder JOBS=... ueberschreibbar

# ---------------------------------------------------------------- Optionen
# "--jobs=4" -> "--jobs 4", "-j4" -> "-j 4"; alles andere bleibt unveraendert.
ARGS=()
for a in "$@"; do
    case $a in
        --*=*)          ARGS+=("${a%%=*}" "${a#*=}") ;;
        -[jcmt][0-9]*)  ARGS+=("${a:0:2}" "${a:2}") ;;
        *)              ARGS+=("$a") ;;
    esac
done
set -- "${ARGS[@]}"

POSITIONAL=()
while (( $# > 0 )); do
    case $1 in
        -j|--jobs|-c|--cores|-m|--mem|-t|--time-limit)
            if (( $# < 2 )); then echo "FEHLER: $1 braucht einen Wert." >&2; exit 1; fi
            case $1 in
                -j|--jobs)       JOBS=$2 ;;
                -c|--cores)      CORES_PER_JOB=$2 ;;
                -m|--mem)        MEM_PER_JOB_GB=$2 ;;
                -t|--time-limit) TIME_LIMIT=$2 ;;
            esac
            shift 2 ;;
        -h|--help) usage; exit 0 ;;
        --)        shift; POSITIONAL+=("$@"); break ;;
        -*)        echo "FEHLER: unbekannte Option '$1' (siehe --help)." >&2; exit 1 ;;
        *)         POSITIONAL+=("$1"); shift ;;
    esac
done
set -- "${POSITIONAL[@]}"

for v in CORES_PER_JOB MEM_PER_JOB_GB TIME_LIMIT; do
    if ! [[ ${!v} =~ ^[1-9][0-9]*$ ]]; then
        echo "FEHLER: $v muss eine positive ganze Zahl sein (ist: '${!v}')." >&2; exit 1
    fi
done
if [[ -n $JOBS ]] && ! [[ $JOBS =~ ^[1-9][0-9]*$ ]]; then
    echo "FEHLER: --jobs muss eine positive ganze Zahl sein (ist: '$JOBS')." >&2; exit 1
fi

if [[ -f "$VENV_DIR/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
fi

# ---------------------------------------------------------------- Task-Liste
TOTAL=$("$PYTHON" main.py --count --instance-dir "$INSTANCE_DIR" 2>/dev/null | tail -n1)
if ! [[ $TOTAL =~ ^[0-9]+$ ]]; then
    echo "FEHLER: 'python main.py --count' lieferte keine Zahl (Ausgabe: '$TOTAL')." >&2
    echo "        Liegt das Script im Projektordner? Ist das venv installiert?" >&2
    exit 1
fi

TASKS=()
if (( $# == 0 )); then
    for ((i = 0; i < TOTAL; i++)); do TASKS+=("$i"); done
else
    # shellcheck disable=SC2068
    set -- ${*//,/ }
    for arg in "$@"; do
        if [[ $arg =~ ^([0-9]+)-([0-9]+)$ ]]; then
            for ((i = BASH_REMATCH[1]; i <= BASH_REMATCH[2]; i++)); do TASKS+=("$i"); done
        elif [[ $arg =~ ^[0-9]+$ ]]; then
            TASKS+=("$arg")
        else
            echo "FEHLER: ungueltiges Argument '$arg' (erlaubt: 5, 3-9, 403,722)." >&2
            exit 1
        fi
    done
fi
for t in "${TASKS[@]}"; do
    if (( t >= TOTAL )); then
        echo "FEHLER: Task-ID $t liegt ausserhalb des Rasters (0..$((TOTAL - 1)))." >&2
        echo "        Liegen alle gewuenschten Methoden in methods/?" >&2
        exit 1
    fi
done

# ---------------------------------------------------------------- Parallelitaet
CORES_TOTAL=$(nproc)
MEM_TOTAL_GB=$(awk '/^MemTotal:/ {printf "%d", $2 / 1024 / 1024}' /proc/meminfo)
JOBS_BY_CPU=$((CORES_TOTAL / CORES_PER_JOB))
JOBS_BY_MEM=$((MEM_TOTAL_GB / MEM_PER_JOB_GB))
JOBS_MAX=$((JOBS_BY_CPU < JOBS_BY_MEM ? JOBS_BY_CPU : JOBS_BY_MEM))
(( JOBS_MAX < 1 )) && JOBS_MAX=1
if [[ -n $JOBS ]]; then
    JOBS_SRC="von dir festgelegt"
else
    JOBS=$DEFAULT_JOBS
    JOBS_SRC="Standard im Script, aendern mit -j"
fi
if (( JOBS > JOBS_BY_CPU )); then
    echo "FEHLER: $JOBS Durchlaeufe x $CORES_PER_JOB Kerne = $((JOBS * CORES_PER_JOB)) Kerne," >&2
    echo "        der Rechner hat aber nur $CORES_TOTAL. Hoechstens -j $JOBS_BY_CPU (oder weniger Kerne je Lauf: -c)." >&2
    exit 1
fi
if (( JOBS > JOBS_BY_MEM )); then
    echo "WARNUNG: $JOBS Durchlaeufe x ${MEM_PER_JOB_GB} GB = $((JOBS * MEM_PER_JOB_GB)) GB, der Rechner hat ${MEM_TOTAL_GB} GB." >&2
    echo "         Wenn alle Laeufe viel Speicher brauchen, kann das System anfangen zu swappen." >&2
fi
if (( JOBS > ${#TASKS[@]} )); then
    JOBS=${#TASKS[@]}    # mehr Slots als Tasks waeren sinnlos
fi

# ---------------------------------------------------------------- RAM-Begrenzung
# 1. Wahl: cgroup ueber systemd (zaehlt den tatsaechlich belegten Speicher;
#    wird ein Lauf zu gross, stirbt NUR dieser, die anderen laufen weiter).
# 2. Wahl: ulimit -v (begrenzt den virtuellen Adressraum, etwas strenger).
if systemd-run --user --scope --quiet -p "MemoryMax=${MEM_PER_JOB_GB}G" true 2>/dev/null; then
    MEM_MODE=cgroup
else
    MEM_MODE=ulimit
fi

LICENSE_ARGS=()
(( SKIP_LICENSE_CHECK )) && LICENSE_ARGS=(--skip-license-check)

mkdir -p logs results/partial

# Interner Arbeitsordner (Warteschlange, Fehlerliste). Bewusst im Projektordner
# und nicht in /tmp: dort kann er unter WSL/Windows verschwinden.
WORKDIR="$PWD/.run_local"
if [[ -f "$WORKDIR/pid" ]] && kill -0 "$(cat "$WORKDIR/pid" 2>/dev/null)" 2>/dev/null; then
    echo "FEHLER: run_local.sh laeuft in diesem Ordner bereits (PID $(cat "$WORKDIR/pid"))." >&2
    echo "        Erst beenden:  pkill -f run_local.sh; pkill -f 'main.py --task-id'" >&2
    exit 1
fi
rm -rf "$WORKDIR"
mkdir -p "$WORKDIR" || { echo "FEHLER: kann $WORKDIR nicht anlegen." >&2; exit 1; }
QUEUE="$WORKDIR/queue"
FAILED="$WORKDIR/failed"
echo $$ > "$WORKDIR/pid"
printf '%s\n' "${TASKS[@]}" > "$QUEUE" || { echo "FEHLER: kann $QUEUE nicht schreiben." >&2; exit 1; }
: > "$FAILED"
[[ -s $QUEUE ]] || { echo "FEHLER: Warteschlange ist leer." >&2; exit 1; }

echo "== run_local.sh =="
echo "Tasks            : ${#TASKS[@]} (von $TOTAL im Raster)"
echo "Parallel         : $JOBS Durchlaeufe gleichzeitig ($JOBS_SRC), je $CORES_PER_JOB Kerne und ${MEM_PER_JOB_GB} GB RAM"
echo "Rechner          : $CORES_TOTAL Kerne, ${MEM_TOTAL_GB} GB RAM"
echo "RAM-Begrenzung   : $MEM_MODE"
echo "Gurobi-Zeitlimit : ${TIME_LIMIT} s pro Durchlauf"
echo "Fortschritt      : tail -f run_local.log   (Strg+C beendet nur die Anzeige, nicht den Lauf)"
echo "Ausgabe je Task  : tail -f logs/local_task<ID>.log"
echo

# ---------------------------------------------------------------- Abbruch (Strg+C)
cleanup() {
    trap - INT TERM
    echo; echo "Abbruch -- beende alle laufenden Durchlaeufe ..." >&2
    rm -rf "$WORKDIR"
    kill 0 2>/dev/null    # gesamte Prozessgruppe: Worker UND ihre Python-Prozesse
}
trap cleanup INT TERM

# ---------------------------------------------------------------- Worker
# Naechste Task-ID aus der gemeinsamen Warteschlange holen (atomar, daher
# bekommt jeder Task genau einmal einen Worker). Dynamisch statt starr
# verteilt, weil die Laufzeiten stark schwanken (Minuten bis ~2,5 Stunden).
next_task() {
    (
        flock 9
        tid=$(head -n1 "$QUEUE")
        if [[ -z $tid ]]; then
            echo -1
        else
            sed -i '1d' "$QUEUE"
            echo "$tid"
        fi
    ) 9> "$QUEUE.lock"
}

run_one() {
    local tid=$1 cpus=$2 log="logs/local_task$1.log"
    local -a limiter=()
    [[ $MEM_MODE == cgroup ]] && limiter=(systemd-run --user --scope --quiet
                                          -p "MemoryMax=${MEM_PER_JOB_GB}G" -p MemorySwapMax=0)
    (
        [[ $MEM_MODE == ulimit ]] && ulimit -Sv $((MEM_PER_JOB_GB * 1024 * 1024))
        # Thread-Variablen begrenzen numpy/BLAS/OpenMP im Python-Teil; Gurobi
        # selbst bekommt seine 4 Threads ueber Params.Threads in den Methoden.
        exec env PYTHONUNBUFFERED=1 \
            OMP_NUM_THREADS="$CORES_PER_JOB" OPENBLAS_NUM_THREADS="$CORES_PER_JOB" \
            MKL_NUM_THREADS="$CORES_PER_JOB" NUMEXPR_NUM_THREADS="$CORES_PER_JOB" \
            VECLIB_MAXIMUM_THREADS="$CORES_PER_JOB" BLIS_NUM_THREADS="$CORES_PER_JOB" \
            "${limiter[@]}" \
            taskset -c "$cpus" \
            "$PYTHON" main.py --task-id "$tid" --instance-dir "$INSTANCE_DIR" \
                --time-limit "$TIME_LIMIT" "${LICENSE_ARGS[@]}"
    ) > "$log" 2>&1
}

worker() {
    local slot=$1
    local first=$((slot * CORES_PER_JOB))
    local cpus="$first-$((first + CORES_PER_JOB - 1))"
    local tid rc t0
    while true; do
        tid=$(next_task)
        [[ $tid == -1 ]] && break
        if ! [[ $tid =~ ^[0-9]+$ ]]; then
            echo "FEHLER: Warteschlange nicht lesbar ($QUEUE) -- Worker $slot bricht ab." >&2
            break
        fi
        t0=$SECONDS
        echo "[$(date +%H:%M:%S)] Task $tid  gestartet  (Kerne $cpus)"
        run_one "$tid" "$cpus"
        rc=$?
        echo "$tid" >> "$WORKDIR/done" 2>/dev/null
        n=$(wc -l < "$WORKDIR/done" 2>/dev/null || echo "?")
        if (( rc == 0 )); then
            echo "[$(date +%H:%M:%S)] Task $tid  fertig      ($((SECONDS - t0)) s)   [$n/${#TASKS[@]} erledigt]"
        else
            echo "[$(date +%H:%M:%S)] Task $tid  FEHLER rc=$rc ($((SECONDS - t0)) s) -> logs/local_task${tid}.log   [$n/${#TASKS[@]} erledigt]" >&2
            echo "$tid" >> "$FAILED"
        fi
    done
}

for ((s = 0; s < JOBS; s++)); do
    worker "$s" &
done
wait

# ---------------------------------------------------------------- Merge
echo
echo "== Merge =="
"$PYTHON" collect_results.py --partial-dir results/partial \
    --instance-dir "$INSTANCE_DIR" --output results/final_results.csv

rc=0
if [[ ! -d $WORKDIR ]]; then
    echo
    echo "WARNUNG: Der Arbeitsordner $WORKDIR wurde waehrend des Laufs geloescht."
    echo "         Es wurden vermutlich NICHT alle Tasks bearbeitet. Pruefe die Ausgabe oben,"
    echo "         starte danach mit den fehlenden IDs neu (Merge-Ausgabe zeigt, was fehlt)."
    exit 1
fi
NFAIL=$(wc -l < "$FAILED")
if (( NFAIL > 0 )); then
    echo
    echo "WARNUNG: $NFAIL Task(s) mit Fehler -- IDs: $(sort -n "$FAILED" | paste -sd, -)"
    echo "         Ursache: logs/local_task<ID>.log. Nach der Korrektur nur diese erneut:"
    echo "         ./run_local.sh $(sort -n "$FAILED" | paste -sd, -)"
    rc=1
fi
rm -rf "$WORKDIR"
exit $rc
