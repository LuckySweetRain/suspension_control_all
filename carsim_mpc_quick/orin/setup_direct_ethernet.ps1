# Run in an elevated Windows PowerShell. Only the dedicated physical Ethernet is changed.
$ErrorActionPreference = 'Stop'
$adapter = Get-NetAdapter -InterfaceIndex 22
if ($adapter.InterfaceDescription -ne 'Realtek Gaming 2.5GbE Family Controller') {
    throw 'Unexpected adapter. Verify the dedicated Ethernet adapter before configuring.'
}
$targetAddress = '192.168.50.1'
$existing = Get-NetIPAddress -InterfaceIndex 22 -AddressFamily IPv4 -ErrorAction SilentlyContinue
if ($existing | Where-Object { $_.IPAddress -ne $targetAddress -and $_.IPAddress -ne '192.168.1.10' -and $_.IPAddress -notlike '169.254.*' }) {
    throw 'Ethernet already has another configured address; leaving it unchanged.'
}
$existing | Where-Object IPAddress -eq '192.168.1.10' | Remove-NetIPAddress -Confirm:$false
if (-not ($existing | Where-Object IPAddress -eq $targetAddress)) {
    New-NetIPAddress -InterfaceIndex 22 -IPAddress $targetAddress -PrefixLength 24 -AddressFamily IPv4 | Out-Null
}
# No default gateway or DNS: Wi-Fi remains the Internet interface.
Get-NetIPAddress -InterfaceIndex 22 -AddressFamily IPv4 | Format-Table IPAddress,PrefixLength
