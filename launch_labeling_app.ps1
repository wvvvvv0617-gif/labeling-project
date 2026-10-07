$python = "C:\Users\qweop\AppData\Local\Programs\Python\Python312\python.exe"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir
& $python "$scriptDir\run_app.py"
