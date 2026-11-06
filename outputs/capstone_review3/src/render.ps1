# Export a .pptx or .docx to PDF with the installed Office app (for visual QA). Usage: render.ps1 <in> <out.pdf>
param([string]$In, [string]$Out)
$In = (Resolve-Path $In).Path
if ($In -like "*.pptx") {
    $app = New-Object -ComObject PowerPoint.Application
    $pres = $app.Presentations.Open($In, $true, $false, $false)   # read-only, no title, no window
    $pres.SaveAs($Out, 32)                                         # ppSaveAsPDF
    $pres.Close(); $app.Quit()
} else {
    $app = New-Object -ComObject Word.Application
    $app.Visible = $false
    $doc = $app.Documents.Open($In, $false, $true)                 # read-only
    $doc.Fields.Update() | Out-Null
    foreach ($toc in $doc.TablesOfContents) { $toc.Update() }
    $doc.ExportAsFixedFormat($Out, 17)                             # wdExportFormatPDF
    $doc.Close($false); $app.Quit()
}
Write-Output "exported $Out"
