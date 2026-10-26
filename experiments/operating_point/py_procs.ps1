Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
  Select-Object ProcessId, ParentProcessId,
    @{n='Start'; e={$_.CreationDate.ToString('HH:mm:ss')}},
    @{n='Cmd'; e={$_.CommandLine.Substring(0, [Math]::Min(140, $_.CommandLine.Length))}} |
  Format-Table -AutoSize | Out-String -Width 260
