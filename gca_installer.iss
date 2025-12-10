; gca_installer.iss

#define MyAppName "Git Commit Assistant"
#define MyAppVersion "1.0"
#define MyAppPublisher "MyCompany"
#define MyAppExeName "gca.exe"

[Setup]
AppId={{A356233B-E436-4022-9226-709724104990}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
; 預設安裝到 C:\Program Files\Git Commit Assistant
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}

OutputBaseFilename=setup_gca

Compression=lzma
SolidCompression=yes

ChangesEnvironment=yes

DisableProgramGroupPage=yes

PrivilegesRequired=admin

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Files]


Source: "dist\gca.exe"; DestDir: "{app}"; Flags: ignoreversion

[Registry]
; to path
Root: HKLM; Subkey: "SYSTEM\CurrentControlSet\Control\Session Manager\Environment"; \
    ValueType: expandsz; ValueName: "Path"; ValueData: "{olddata};{app}"; \
    Check: NeedsAddPath(ExpandConstant('{app}'))

[Code]

function NeedsAddPath(Param: string): boolean;
var
  OrigPath: string;
begin
  if not RegQueryStringValue(HKEY_LOCAL_MACHINE,
    'SYSTEM\CurrentControlSet\Control\Session Manager\Environment',
    'Path', OrigPath)
  then begin
    Result := True;
    exit;
  end;
  Result := Pos(';' + UpperCase(Param) + ';', ';' + UpperCase(OrigPath) + ';') = 0;
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  BinPath, PathStr, NewPathStr: string;
  P, P_end: Integer;
begin
  if CurUninstallStep = usUninstall then begin
    BinPath := UpperCase(ExpandConstant('{app}'));
    if RegQueryStringValue(HKEY_LOCAL_MACHINE, 'SYSTEM\CurrentControlSet\Control\Session Manager\Environment', 'Path', PathStr) then
    begin
      if Pos(BinPath, UpperCase(PathStr)) > 0 then begin

      end;
    end;
  end;
end;