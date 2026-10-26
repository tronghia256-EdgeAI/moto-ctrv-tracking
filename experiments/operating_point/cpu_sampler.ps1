# 1 Hz sampler of \Processor Information(_Total)\% Processor Performance for the pre-registered
# latency validity rule (evidence.md, ce8f3d5). Stamps come from this machine's system clock,
# the same clock as the trial files' mtimes. Stops when the sentinel file exists.
#   powershell -NoProfile -File experiments/operating_point/cpu_sampler.ps1 <out.csv> <sentinel>
param([string]$Out, [string]$Stop)
"unix_ms,local_time,processor_performance" | Out-File -FilePath $Out -Encoding ascii
while (-not (Test-Path $Stop)) {
    $s = (Get-Counter '\Processor Information(_Total)\% Processor Performance' -SampleInterval 1 -MaxSamples 1).CounterSamples[0]
    $t = [DateTimeOffset]::new($s.Timestamp).ToUnixTimeMilliseconds()
    "$t,$($s.Timestamp.ToString('o')),$([math]::Round($s.CookedValue, 2))" | Out-File -FilePath $Out -Append -Encoding ascii
}
