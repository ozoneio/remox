param(
    [Parameter(Mandatory = $true)][int]$ProcessId,
    [Parameter(Mandatory = $true)][string]$FilePath,
    [Parameter(Mandatory = $true)][string]$FileNameAutomationId
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
Add-Type -AssemblyName UIAutomationClient
Add-Type -AssemblyName UIAutomationTypes
Add-Type -AssemblyName UIAutomationClientsideProviders
$desktop = [System.Windows.Automation.AutomationElement]::RootElement
[System.Windows.Automation.ClientSettings]::RegisterClientSideProviders(
    [UIAutomationClientsideProviders.UIAutomationClientSideProviders]::ClientSideProviderDescriptionTable
)
Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public static class RemoxDialogNative {
    [DllImport("user32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    public static extern bool PostMessage(IntPtr window, uint message, IntPtr wParam, IntPtr lParam);
}
'@

function Get-DialogControls {
    param([Parameter(Mandatory = $true)][System.Windows.Automation.AutomationElement]$Dialog)

    $controls = $Dialog.FindAll(
        [System.Windows.Automation.TreeScope]::Descendants,
        [System.Windows.Automation.Condition]::TrueCondition
    )
    return ($controls | Where-Object {
        $_.Current.AutomationId -in @('1148', '1', '2', 'FileNameControlHost') -or
        $_.Current.ControlType -eq [System.Windows.Automation.ControlType]::Edit -or
        $_.Current.ClassName -eq 'Edit'
    } | ForEach-Object {
        [PSCustomObject]@{
            automationId = $_.Current.AutomationId
            controlType = $_.Current.ControlType.ProgrammaticName
            className = $_.Current.ClassName
        }
    } | ConvertTo-Json -Compress)
}

$dialogCondition = [System.Windows.Automation.AndCondition]::new(
    [System.Windows.Automation.PropertyCondition]::new(
        [System.Windows.Automation.AutomationElement]::ProcessIdProperty, $ProcessId
    ),
    [System.Windows.Automation.PropertyCondition]::new(
        [System.Windows.Automation.AutomationElement]::ClassNameProperty, '#32770'
    )
)
$deadline = [DateTime]::UtcNow.AddSeconds(15)
$dialogs = $desktop.FindAll([System.Windows.Automation.TreeScope]::Children, $dialogCondition)
while ($dialogs.Count -eq 0 -and [DateTime]::UtcNow -lt $deadline) {
    Start-Sleep -Milliseconds 100
    $dialogs = $desktop.FindAll([System.Windows.Automation.TreeScope]::Children, $dialogCondition)
}
if ($dialogs.Count -ne 1) {
    throw "Expected one native file dialog for process $ProcessId; found $($dialogs.Count)."
}
$dialog = $dialogs[0]
$fileNameCondition = [System.Windows.Automation.AndCondition]::new(
    [System.Windows.Automation.PropertyCondition]::new(
        [System.Windows.Automation.AutomationElement]::AutomationIdProperty, $FileNameAutomationId
    ),
    [System.Windows.Automation.PropertyCondition]::new(
        [System.Windows.Automation.AutomationElement]::ControlTypeProperty,
        [System.Windows.Automation.ControlType]::Edit
    )
)
$fileNames = $dialog.FindAll([System.Windows.Automation.TreeScope]::Descendants, $fileNameCondition)
if ($fileNames.Count -ne 1) {
    throw "Expected filename edit AutomationId $FileNameAutomationId; found $($fileNames.Count). Controls: $(Get-DialogControls -Dialog $dialog)"
}
$valuePattern = [System.Windows.Automation.ValuePattern]$fileNames[0].GetCurrentPattern(
    [System.Windows.Automation.ValuePattern]::Pattern
)
$valuePattern.SetValue($FilePath)

$confirmCondition = [System.Windows.Automation.AndCondition]::new(
    [System.Windows.Automation.PropertyCondition]::new(
        [System.Windows.Automation.AutomationElement]::AutomationIdProperty, '1'
    ),
    [System.Windows.Automation.PropertyCondition]::new(
        [System.Windows.Automation.AutomationElement]::ClassNameProperty, 'Button'
    )
)
$confirmButtons = $dialog.FindAll([System.Windows.Automation.TreeScope]::Descendants, $confirmCondition)
if ($confirmButtons.Count -ne 1) {
    throw "Expected confirm button AutomationId 1; found $($confirmButtons.Count). Controls: $(Get-DialogControls -Dialog $dialog)"
}
# The common dialog's split confirm button has no managed InvokePattern provider.
# Address the exact UIA-selected native control with the documented BM_CLICK message.
$confirmHandle = [IntPtr]$confirmButtons[0].Current.NativeWindowHandle
if ($confirmHandle -eq [IntPtr]::Zero) { throw 'Native confirm button has no window handle.' }
if (-not [RemoxDialogNative]::PostMessage($confirmHandle, 0x00F5, [IntPtr]::Zero, [IntPtr]::Zero)) {
    throw [System.ComponentModel.Win32Exception]::new([Runtime.InteropServices.Marshal]::GetLastWin32Error())
}
@{ processId = $ProcessId; filePath = $FilePath; confirmed = $true } | ConvertTo-Json -Compress
