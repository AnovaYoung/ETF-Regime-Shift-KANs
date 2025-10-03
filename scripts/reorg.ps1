\
# SAFE REORG of existing messy structure into the new standardized layout.
# Run from the repo root. Review output first; remove -WhatIf to actually move.

$raw = "data/raw/fundflow"
$newProcessed = "data/processed"
$newInterim = "data/interim"
$kanFolder = "KANs"

# Create folders
New-Item -ItemType Directory -Force -Path $raw,$newProcessed,$newInterim | Out-Null

# Move per-ticker *_fundflow dirs
Get-ChildItem $kanFolder -Directory -Filter "*_fundflow" |
    ForEach-Object {
        $target = Join-Path $raw $_.Name
        Move-Item -Path $_.FullName -Destination $target -WhatIf
    }

# Move known parquet artifacts
Get-ChildItem $kanFolder -File -Filter "*.parquet" |
    ForEach-Object {
        Move-Item -Path $_.FullName -Destination $newProcessed -WhatIf
    }

Write-Host "Dry-run complete. If the moves look good, remove -WhatIf and run again."
