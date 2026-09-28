# Keep Windows from sleeping (the display may still turn off) until process -WaitPid exits.
# Uses SetThreadExecutionState, so no power setting is changed; the request ends with this script.
#
#   powershell -ExecutionPolicy Bypass -File scripts\keep_awake.ps1 -WaitPid 1234
param([Parameter(Mandatory = $true)][int]$WaitPid)
Add-Type -Namespace Win32 -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
$ES_CONTINUOUS = [uint32]"0x80000000"
$ES_SYSTEM_REQUIRED = [uint32]"0x00000001"
[void][Win32.Power]::SetThreadExecutionState($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED)
Write-Output "$(Get-Date -Format 'HH:mm:ss') keeping the system awake until PID $WaitPid exits"
try {
    Wait-Process -Id $WaitPid -ErrorAction SilentlyContinue
} finally {
    [void][Win32.Power]::SetThreadExecutionState($ES_CONTINUOUS)
    Write-Output "$(Get-Date -Format 'HH:mm:ss') released; normal sleep settings apply again"
}
