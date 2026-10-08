$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$matcherPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $matcherPython)) {
    throw 'Missing local environment. Create .venv and install requirements.txt first.'
}
# Use the generated demo catalog unless the caller selects another local catalog.
if ([string]::IsNullOrWhiteSpace($env:BBS_CATALOG_DB)) {
    $env:BBS_CATALOG_DB = Join-Path $PSScriptRoot 'demo_catalog.db'
}
# Normal app use keeps previews in memory; no persistent upload/image traces.
$env:BBS_MATCH_TRACE_DIR = ''
& $matcherPython -X utf8 -m streamlit run (Join-Path $PSScriptRoot 'app.py') @args
exit $LASTEXITCODE
