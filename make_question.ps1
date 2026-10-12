# Creates a caller question WAV (16 kHz, mono, 16-bit) from text using the Windows built-in voice.
# Usage:
#   powershell -ExecutionPolicy Bypass -File .\make_question.ps1 -Text "I want to book an appointment" -Name booking
# Output: uploads\<Name>.wav

param(
    [Parameter(Mandatory = $true)][string]$Text,
    [string]$Name = "question"
)

Set-Location $PSScriptRoot
Add-Type -AssemblyName System.Speech
$out = Join-Path $PSScriptRoot "uploads\$Name.wav"
$fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000,
    [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono)
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
$s.SetOutputToWaveFile($out, $fmt)
$s.Speak($Text)
$s.Dispose()
Write-Host "Created $out"
Write-Host "Use it with: --question uploads/$Name.wav"
