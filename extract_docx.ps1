$docx = "c:\Users\Pichau\Downloads\Projeto_Final_IA-6a11efa6a363d.docx"
$out = "c:\Users\Pichau\Documents\ProjetoIA\docx_extract.txt"
$temp = Join-Path $env:TEMP ("docx_extract_" + [guid]::NewGuid().ToString())
New-Item -ItemType Directory -Path $temp -Force | Out-Null
$zipPath = Join-Path $temp "doc.zip"
Copy-Item -LiteralPath $docx -Destination $zipPath
Expand-Archive -LiteralPath $zipPath -DestinationPath (Join-Path $temp "unzipped") -Force
$xmlPath = Join-Path $temp "unzipped\word\document.xml"
[xml]$xmlDoc = Get-Content -LiteralPath $xmlPath -Raw -Encoding UTF8
$nsMgr = New-Object System.Xml.XmlNamespaceManager($xmlDoc.NameTable)
$nsMgr.AddNamespace("w", "http://schemas.openxmlformats.org/wordprocessingml/2006/main")
$sb = New-Object System.Text.StringBuilder
$body = $xmlDoc.SelectSingleNode("//w:body", $nsMgr)
foreach ($node in $body.ChildNodes) {
    $local = $node.LocalName
    if ($local -eq "p") {
        $textNodes = $node.SelectNodes(".//w:t", $nsMgr)
        $line = ($textNodes | ForEach-Object { $_.InnerText }) -join ""
        [void]$sb.AppendLine($line)
    }
    elseif ($local -eq "tbl") {
        $rows = $node.SelectNodes(".//w:tr", $nsMgr)
        foreach ($row in $rows) {
            $cells = @()
            foreach ($tc in $row.SelectNodes("./w:tc", $nsMgr)) {
                $cellText = @()
                foreach ($p in $tc.SelectNodes(".//w:p", $nsMgr)) {
                    $tns = $p.SelectNodes(".//w:t", $nsMgr)
                    $pt = ($tns | ForEach-Object { $_.InnerText }) -join ""
                    if ($pt) { $cellText += $pt }
                }
                $cells += ($cellText -join " ")
            }
            [void]$sb.AppendLine(($cells -join "`t"))
        }
    }
}
$text = $sb.ToString()
Set-Content -LiteralPath $out -Value $text -Encoding UTF8
Write-Output ("Saved " + $text.Length + " chars to " + $out)
