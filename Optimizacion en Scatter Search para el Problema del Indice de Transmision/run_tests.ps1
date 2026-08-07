# Forzar UTF-8 de punta a punta para que los acentos/enie no se corrompan
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding           = [System.Text.Encoding]::UTF8

$BASE   = (Get-Location).Path
$SCRIPT = Join-Path $BASE "Optimizacion_en_Scatter_Search_para_el_Problema_del_Indice_de_Transmision.py"
$GRAFOS = Join-Path $BASE "test_graphs"
$LOG_DIR = Join-Path $BASE "logs_multi"     # carpeta NUEVA para las corridas multiples
$RESUMEN = Join-Path $LOG_DIR "resumen.txt"

Write-Host "Directorio base: $BASE"
if (-Not (Test-Path $SCRIPT)) {
    Write-Host "ERROR: No se encontro el archivo Python en la misma carpeta que este script."
    exit 1
}
New-Item -ItemType Directory -Force -Path $LOG_DIR | Out-Null
"Resultados Scatter Search - corridas multiples - $(Get-Date)" | Out-File -FilePath $RESUMEN -Encoding utf8
"========================================" | Out-File -FilePath $RESUMEN -Append -Encoding utf8

# Esquema graduado: mas corridas donde es barato, menos donde es caro
$RUNS = @{ "small" = 8; "medium" = 4; "large" = 2 }

# Casos que corres TU personalmente (se excluyen aqui)
$EXCLUIR = @(
    "cycle_graph_large_(3)",       # ciclo n=300
    "hypercube_graph_large_(3)"    # Q9
)

$tipos = @("star_graph", "cycle_graph", "wheel_graph", "hypercube_graph", "de_bruijn_graph")
$MAX_NODOS       = 600   # limite general para grandes
$MAX_NODOS_CICLO = 300

$totalCorridas = 0
foreach ($tipo in $tipos) {
    foreach ($tam in @("small", "medium", "large")) {
        $carpeta = Join-Path $GRAFOS "$tipo\$tam"
        if (-Not (Test-Path $carpeta)) { continue }
        $n = $RUNS[$tam]
        foreach ($f in Get-ChildItem "$carpeta\*.txt") {
            $nombre = $f.BaseName

            # --- exclusiones ---
            if ($EXCLUIR -contains $nombre) {
                Write-Host "Excluido (lo corres tu): $nombre"; continue
            }
            if ($tipo -eq "de_bruijn_graph" -and $tam -eq "large") {
                Write-Host "Excluido De Bruijn grande (lo corres tu): $nombre"; continue
            }

            # --- cap de nodos para grandes ---
            if ($tam -eq "large") {
                $nodos = (Get-Content $f.FullName | Measure-Object -Line).Lines - 1
                $limite = if ($tipo -eq "cycle_graph") { $MAX_NODOS_CICLO } else { $MAX_NODOS }
                if ($nodos -gt $limite) {
                    Write-Host "Omitiendo $nombre ($nodos nodos > $limite)"; continue
                }
            }

            # --- N corridas de este caso (los grandes incluidos van 2 veces) ---
            for ($i = 1; $i -le $n; $i++) {
                $logPath = Join-Path $LOG_DIR "${nombre}_r$i.log"
                Write-Host "Corriendo $nombre  (corrida $i de $n)"
                & python -X utf8 "$SCRIPT" "$($f.FullName)" 2>&1 | Out-File -FilePath $logPath -Encoding utf8
                $idx    = Select-String -Path $logPath -Pattern "aproximado \(real\)" | Select-Object -First 1 -ExpandProperty Line
                $tiempo = Select-String -Path $logPath -Pattern "Tiempo de ejecuci"   | Select-Object -First 1 -ExpandProperty Line
                "${nombre}_r${i}: $idx | $tiempo" | Out-File -FilePath $RESUMEN -Append -Encoding utf8
                Write-Host "  -> $idx | $tiempo"
                $totalCorridas++
            }
        }
    }
}

"========================================" | Out-File -FilePath $RESUMEN -Append -Encoding utf8
Write-Host ""
Write-Host "Total de corridas realizadas: $totalCorridas"
Write-Host "Logs en: $LOG_DIR   (cada caso como <nombre>_r1.log, _r2.log, ... con indice y tiempo)"
