$docx = "c:\Users\Pichau\Downloads\Projeto_Final_IA-6a11efa6a363d.docx"
$out = "c:\Users\Pichau\Documents\ProjetoIA\docx_extract.txt"
$temp = Join-Path $env:TEMP ("docx_extract2_" + [guid]::NewGuid().ToString())
New-Item -ItemType Directory -Path $temp -Force | Out-Null
Copy-Item -LiteralPath $docx -Destination (Join-Path $temp "doc.zip")
Expand-Archive -LiteralPath (Join-Path $temp "doc.zip") -DestinationPath (Join-Path $temp "u") -Force
$xmlPath = Join-Path $temp "u\word\document.xml"
[xml]$xmlDoc = Get-Content -LiteralPath $xmlPath -Raw -Encoding UTF8
$nsMgr = New-Object System.Xml.XmlNamespaceManager($xmlDoc.NameTable)
$nsMgr.AddNamespace("w", "http://schemas.openxmlformats.org/wordprocessingml/2006/main")
$paras = $xmlDoc.SelectNodes("//w:body//w:p", $nsMgr)
$lines = New-Object System.Collections.Generic.List[string]
foreach ($p in $paras) {
    $tns = $p.SelectNodes(".//w:t", $nsMgr)
    $line = ($tns | ForEach-Object { $_.'#text' }) -join ""
    if ($line -match '\S') { [void]$lines.Add($line) } else { [void]$lines.Add("") }
}
$text = ($lines -join "`n").TrimEnd()
$utf8 = New-Object System.Text.UTF8Encoding $false
[System.IO.File]::WriteAllText($out, $text, $utf8)
Write-Output "Paras: $($paras.Count) Chars: $($text.Length)"
Write-Output $text
