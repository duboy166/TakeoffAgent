; Inno Setup Script for Takeoff Agent
; Requires Inno Setup 6.x (https://jrsoftware.org/isinfo.php)
;
; This script creates a Windows installer that:
;   - Installs to Program Files
;   - Creates Start Menu shortcuts
;   - Optionally creates Desktop shortcut
;   - Registers uninstaller
;   - Uses LZMA2 compression
;
; Build command:
;   iscc scripts\installer.iss

#define MyAppName "Takeoff Agent"
#define MyAppVersion "1.0.0"
#define MyAppPublisher "Takeoff Agent"
#define MyAppExeName "TakeoffAgent.exe"
#define MyAppURL "https://github.com/your-repo/takeoff-agent"

[Setup]
; Application identification
AppId={{E8B7A5F2-4C3D-4F6E-9A8B-1C2D3E4F5A6B}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppVerName={#MyAppName} {#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}

; Installation directories
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
AllowNoIcons=yes

; Output settings
OutputDir=..\dist
OutputBaseFilename=TakeoffAgent_Setup

; Compression (LZMA2 for best compression)
Compression=lzma2/ultra64
SolidCompression=yes
LZMANumBlockThreads=4

; Elevation and permissions
PrivilegesRequired=admin
PrivilegesRequiredOverridesAllowed=dialog

; Icons (optional - will be skipped if not found)
SetupIconFile=..\assets\icon.ico
UninstallDisplayIcon={app}\{#MyAppExeName}

; Appearance
WizardStyle=modern
WizardSizePercent=100

; Architecture (64-bit only)
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible

; Misc
DisableProgramGroupPage=yes
DisableReadyPage=no
DisableFinishedPage=no

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
; Main application directory (recursive copy of entire dist\TakeoffAgent folder)
Source: "..\dist\TakeoffAgent\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
; Start Menu
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\{cm:UninstallProgram,{#MyAppName}}"; Filename: "{uninstallexe}"
; Desktop (optional)
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
; Option to launch app after installation
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
; Clean up any files created during use
Type: filesandordirs; Name: "{app}\logs"
Type: filesandordirs; Name: "{app}\temp"
Type: dirifempty; Name: "{app}"

[Code]
// Custom code to show installation progress
procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep = ssInstall then
  begin
    // Installing
    WizardForm.StatusLabel.Caption := 'Installing Takeoff Agent...';
  end;
end;

// Check for sufficient disk space (approximately 600MB for full installation)
function InitializeSetup(): Boolean;
var
  FreeMB: Cardinal;
begin
  Result := True;

  // Check disk space (600MB minimum)
  FreeMB := GetSpaceOnDisk(ExpandConstant('{autopf}'), True, True);
  if FreeMB < 600 then
  begin
    MsgBox('Takeoff Agent requires at least 600 MB of free disk space.' + #13#10 +
           'Please free up some space and try again.', mbError, MB_OK);
    Result := False;
  end;
end;
