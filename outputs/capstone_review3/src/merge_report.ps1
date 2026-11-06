# Merge front matter + body with Word, keeping the body's header/footer and restarting page numbers at Chapter 1.
# Usage: merge_report.ps1 <front.docx> <body.docx> <out.docx>
param([string]$Front, [string]$Body, [string]$Out)
$Front = (Resolve-Path $Front).Path
$Body = (Resolve-Path $Body).Path
$w = New-Object -ComObject Word.Application
$w.Visible = $false
$w.DisplayAlerts = 0
try {
    $doc = $w.Documents.Open($Body, $false, $false)
    $doc.Range(0, 0).InsertBreak(2)                  # wdSectionBreakNextPage -> new empty section 1
    $s1 = $doc.Sections.Item(1)
    $s2 = $doc.Sections.Item(2)
    foreach ($i in 1..3) {                           # primary, first page, even
        $s2.Headers.Item($i).LinkToPrevious = $false
        $s2.Footers.Item($i).LinkToPrevious = $false
    }
    foreach ($i in 1..3) {
        foreach ($hf in @($s1.Headers.Item($i), $s1.Footers.Item($i))) {
            while ($hf.Range.Tables.Count -gt 0) { $hf.Range.Tables.Item(1).Delete() }
            while ($hf.Range.InlineShapes.Count -gt 0) { $hf.Range.InlineShapes.Item(1).Delete() }
            $hf.Range.Text = ""
            $hf.Range.ParagraphFormat.Borders.Enable = 0
        }
    }
    $s2.Footers.Item(1).PageNumbers.RestartNumberingAtSection = $true
    $s2.Footers.Item(1).PageNumbers.StartingNumber = 1
    $s1.PageSetup.DifferentFirstPageHeaderFooter = $false
    $doc.Range(0, 0).InsertFile($Front)
    # drop the empty paragraph left before the section break, if any
    $doc.SaveAs2($Out, 16)                           # wdFormatDocumentDefault (.docx)
    $doc.Close($false)
    Write-Output "merged -> $Out"
} finally {
    $w.Quit()
}
