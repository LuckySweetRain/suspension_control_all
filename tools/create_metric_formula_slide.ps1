$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

function RgbInt([int]$r, [int]$g, [int]$b) {
    return $r + ($g * 256) + ($b * 65536)
}

function Invoke-Retry([scriptblock]$Action, [int]$Attempts = 10, [int]$DelayMs = 500) {
    $last = $null
    for ($i = 0; $i -lt $Attempts; $i++) {
        try {
            return & $Action
        } catch {
            $last = $_
            Start-Sleep -Milliseconds $DelayMs
        }
    }
    throw $last
}

function Add-TextBox($slide, [string]$text, [double]$x, [double]$y, [double]$w, [double]$h, [int]$size, [int]$color, [bool]$bold = $false, [string]$font = "Microsoft YaHei") {
    $shape = $slide.Shapes.AddTextbox(1, $x, $y, $w, $h)
    $shape.TextFrame2.MarginLeft = 0
    $shape.TextFrame2.MarginRight = 0
    $shape.TextFrame2.MarginTop = 0
    $shape.TextFrame2.MarginBottom = 0
    $shape.TextFrame2.TextRange.Text = $text
    $shape.TextFrame2.TextRange.Font.Name = $font
    $shape.TextFrame2.TextRange.Font.NameFarEast = $font
    $shape.TextFrame2.TextRange.Font.Size = $size
    $shape.TextFrame2.TextRange.Font.Fill.ForeColor.RGB = $color
    if ($bold) {
        $shape.TextFrame2.TextRange.Font.Bold = -1
    }
    return $shape
}

function Add-Card($slide, [double]$x, [double]$y, [double]$w, [double]$h, [int]$fill, [int]$line) {
    $shape = $slide.Shapes.AddShape(5, $x, $y, $w, $h)
    $shape.Fill.ForeColor.RGB = $fill
    $shape.Line.ForeColor.RGB = $line
    $shape.Line.Weight = 0.75
    return $shape
}

function Paste-Equation($slide, [string]$formula, [double]$x, [double]$y, [double]$w, [double]$h, [double]$fontSize, [int]$color) {
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    $doc = $word.Documents.Add()
    try {
        $range = $doc.Range(0, 0)
        $range.Text = $formula
        $eqRange = $doc.Range(0, $formula.Length)
        $eqRange.Font.Size = $fontSize
        [void]$doc.OMaths.Add($eqRange)
        $omath = $doc.OMaths.Item(1)
        $omath.BuildUp()
        $omath.Range.Font.Size = $fontSize
        $omath.Range.Copy()
        Start-Sleep -Milliseconds 700
        $shapeRange = Invoke-Retry { $slide.Shapes.PasteSpecial(0) } 12 450
        $shape = Invoke-Retry {
            try {
                return $shapeRange.Item(1)
            } catch {
                try {
                    return $shapeRange[1]
                } catch {
                    return $shapeRange
                }
            }
        } 12 450
        Invoke-Retry { $shape.Left = $x; $shape.Top = $y; $shape.Width = $w; $shape.Height = $h; $null } 12 450 | Out-Null
        try {
            $shape.TextFrame2.MarginLeft = 0
            $shape.TextFrame2.MarginRight = 0
            $shape.TextFrame2.MarginTop = 0
            $shape.TextFrame2.MarginBottom = 0
            $shape.TextFrame2.TextRange.Font.Size = $fontSize
            $shape.TextFrame2.TextRange.Font.Fill.ForeColor.RGB = $color
        } catch {
            # Some pasted equation shapes reject text-frame edits; position/size above is sufficient.
        }
        return $shape
    } finally {
        $doc.Close($false)
        $word.Quit()
        [System.Runtime.InteropServices.Marshal]::ReleaseComObject($doc) | Out-Null
        [System.Runtime.InteropServices.Marshal]::ReleaseComObject($word) | Out-Null
    }
}

$outDir = "E:\Desktop\研究生中期"
$outPpt = Join-Path $outDir "评价指标与统一优化目标-公式页.pptx"
$previewDir = Join-Path $outDir "评价指标与统一优化目标-公式页预览"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null
if (Test-Path $outPpt) { Remove-Item -LiteralPath $outPpt -Force }
if (Test-Path $previewDir) { Remove-Item -LiteralPath $previewDir -Recurse -Force }

$primary = RgbInt 3 29 64
$blue = RgbInt 68 114 196
$text = RgbInt 29 36 48
$muted = RgbInt 75 85 99
$tertiary = RgbInt 122 134 154
$border = RgbInt 217 226 239
$grid = RgbInt 232 238 247
$lightBlue = RgbInt 234 242 255
$paleRed = RgbInt 255 241 242
$success = RgbInt 15 118 110

$pp = New-Object -ComObject PowerPoint.Application
$pp.Visible = -1
$pres = $pp.Presentations.Add()
$pres.PageSetup.SlideWidth = 960
$pres.PageSetup.SlideHeight = 540
$slide = $pres.Slides.Add(1, 12)

# Background and header
$bg = $slide.Shapes.AddShape(1, 0, 0, 960, 540)
$bg.Fill.ForeColor.RGB = RgbInt 255 255 255
$bg.Line.Visible = 0
Add-TextBox $slide "评价指标与统一优化目标" 42 32 520 34 26 $primary $true "Microsoft YaHei" | Out-Null
Add-TextBox $slide "从舒适性、安全约束到 MPC/RL 统一代价" 42 67 520 20 11 $tertiary $false "Microsoft YaHei" | Out-Null
$line = $slide.Shapes.AddLine(42, 94, 918, 94)
$line.Line.ForeColor.RGB = $border
$line.Line.Weight = 0.75

# Left metric system panel
Add-Card $slide 48 116 302 312 (RgbInt 255 255 255) $border | Out-Null
$accent = $slide.Shapes.AddShape(1, 48, 116, 5, 312)
$accent.Fill.ForeColor.RGB = $primary
$accent.Line.Visible = 0
Add-TextBox $slide "评价指标体系" 72 136 160 24 17 $primary $true "Microsoft YaHei" | Out-Null
Add-TextBox $slide "主动悬架控制需要同时考虑舒适性、安全性和执行器负担。" 72 164 240 34 10 $muted $false "Microsoft YaHei" | Out-Null

$items = @(
    @("舒适性", "车身垂向加速度 RMS；俯仰角加速度 RMS", $lightBlue, $blue),
    @("安全性", "悬架动行程峰值；轮胎动载荷 RMS", (RgbInt 243 246 250), $primary),
    @("执行器需求", "控制力峰值；控制能耗", $paleRed, (RgbInt 185 28 28)),
    @("约束条件", "悬架行程、轮胎动载荷、执行器力限幅", (RgbInt 243 246 250), $success)
)
$y = 216
foreach ($it in $items) {
    $chip = $slide.Shapes.AddShape(5, 72, $y, 82, 26)
    $chip.Fill.ForeColor.RGB = $it[2]
    $chip.Line.ForeColor.RGB = $border
    $chip.Line.Weight = 0.5
    Add-TextBox $slide $it[0] 88 $($y + 6) 54 12 9 $it[3] $true "Microsoft YaHei" | Out-Null
    Add-TextBox $slide $it[1] 166 $($y + 4) 142 20 9 $text $false "Microsoft YaHei" | Out-Null
    $y += 52
}

Add-TextBox $slide "约束形式：" 72 393 70 16 10 $primary $true "Microsoft YaHei" | Out-Null
Add-TextBox $slide "|Δy_f|≤Δy_{f,max}，|Δy_r|≤Δy_{r,max}，|u_af|,|u_ar|≤u_max" 140 394 190 20 8 $muted $false "Cambria Math" | Out-Null

# Right formula panels
Add-Card $slide 382 116 530 112 $lightBlue $border | Out-Null
Add-TextBox $slide "RMS 定义" 405 132 120 22 16 $primary $true "Microsoft YaHei" | Out-Null
Add-TextBox $slide "对任一信号 q(t)，用稳定段内均方根衡量振动强度。" 405 158 330 18 10 $muted $false "Microsoft YaHei" | Out-Null
Paste-Equation $slide "RMS(q)=√(1/(T_2-T_1) ∫_(T_1)^(T_2) q^2(t)ⅆt)" 430 178 440 46 18 $primary | Out-Null
Add-TextBox $slide "其中 T₁ 为稳定段起始时间。" 725 137 150 16 9 $tertiary $false "Microsoft YaHei" | Out-Null

Add-Card $slide 382 250 530 178 (RgbInt 255 255 255) $border | Out-Null
Add-TextBox $slide "MPC 与强化学习统一多目标代价" 405 266 300 22 16 $primary $true "Microsoft YaHei" | Out-Null
Add-TextBox $slide "各指标先归一化，再按权重合成为瞬时代价。" 405 291 260 16 10 $muted $false "Microsoft YaHei" | Out-Null
Paste-Equation $slide "J=1/T ∫_0^T [w_1 (z̈_b/a_(z,max))^2+w_2 (θ̈_b/α_(θ,max))^2" 410 310 470 30 12.5 $primary | Out-Null
Paste-Equation $slide "+w_3 ((Δy_f/Δy_max)^2+(Δy_r/Δy_max)^2)+w_4 (((F_tf+F_bf)/F_(zf,0))^2+((F_tr+F_br)/F_(zr,0))^2)" 410 342 470 36 11.2 $primary | Out-Null
Paste-Equation $slide "+w_5 ‖u/u_max‖_2^2] ⅆt" 410 382 230 28 13 $primary | Out-Null
Add-TextBox $slide "归一化因子：a_z,max，α_θ,max，Δy_max，F_zf,0，F_zr,0，u_max" 640 386 220 28 8.5 $tertiary $false "Microsoft YaHei" | Out-Null

# Bottom reward strip
Add-Card $slide 48 452 864 54 (RgbInt 243 246 250) $border | Out-Null
Add-TextBox $slide "强化学习奖励" 72 468 110 18 14 $primary $true "Microsoft YaHei" | Out-Null
Paste-Equation $slide "r_t=-ℓ_t" 178 464 104 30 18 $primary | Out-Null
Add-TextBox $slide "奖励取负代价，使智能体在降低车身振动的同时，避免悬架击穿、轮胎跳动和执行器饱和。" 300 469 540 20 12 $text $false "Microsoft YaHei" | Out-Null
Add-TextBox $slide "评价指标 → 约束归一化 → 多目标代价 → PPO/MPC 统一优化" 300 489 520 14 9 $tertiary $false "Microsoft YaHei" | Out-Null

$pres.SaveAs($outPpt)
$pres.Export($previewDir, "PNG", 1280, 720)
$pres.Close()
$pp.Quit()
[System.Runtime.InteropServices.Marshal]::ReleaseComObject($slide) | Out-Null
[System.Runtime.InteropServices.Marshal]::ReleaseComObject($pres) | Out-Null
[System.Runtime.InteropServices.Marshal]::ReleaseComObject($pp) | Out-Null

[pscustomobject]@{
    PPTX = $outPpt
    PreviewDir = $previewDir
}
