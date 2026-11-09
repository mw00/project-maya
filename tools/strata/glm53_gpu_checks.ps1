# tools/strata/glm53_gpu_checks.ps1 - glm53_gpu_checks.sh's identity checks on Windows (HIP or CUDA builds): the glm-dsa
# (full GLM-5.3) port must not move a bit of GLM-5.3-Flash or DeepSeek.  Nothing else may use the GPU meanwhile: the
# 8080 dashboard holds it (stop it first; -Force skips that check), and two engines at once fight for memory.
#
#   glm53_gpu_checks.ps1 flash-ab -FlashPack DIR -AbTokens F [-Lane off|default|both] [-Runs N] [-Base B] [-Port P]
#   glm53_gpu_checks.ps1 ds4-ab   -Ds4Pack DIR -Ds4Ids F [-Ds4Args '...'] [-Runs N] [-NoDump] [-Tag T] [-Base B] [-Port P]
#
# B and P are build folders with maya.exe / maya-ds4.exe: B a release build (v1.1.56), P this tree's (default
# build-hip, else build).  -LibDir (default $env:HIP_PATH\bin) goes first on PATH for the HIP runtime.  Results go to
# -Out (default %TEMP%\glm53-gpu-<date>); every check prints PASS / FAIL, the last line is the summary (exit 1 on a FAIL).
# flash-ab: STRATA_GLM_SCORE log-softmax dumps over the token file (-Ctx + 256 ids at most; a Flash-vocab document)
#   and -MaxNew greedy tokens after its first -Ctx ids, in the order base, port, base, port, ... (-Runs rounds).  Every
#   run must equal the first byte for byte: the dumps, and the greedy runs' whole output (Answer: the INFO / VRAM lines'
#   expert tiers and memory layout, the tokens, the DONE line's counts, finish and drafting counters), masked only where
#   it follows the clock or the memory free at load (the DONE line's wall-clock fields, the VRAM lines' free= and
#   window=).  The greedy runs fix the MTP draft length (STRATA_GLM_MTP_ADAPT=0) so that the drafting counters repeat.
#   A run that exits non-zero, prints ERR or ends without a DONE line counting its tokens is a FAIL by itself.
#   -Lane off (default): STRATA_GLM_CPU_LANE=0 STRATA_GLM_PREFILL_CPU=0.  -Lane default: the machine's own defaults
#   (e.g. the one-GPU MTP decode on gfx115x, its draft length fixed as above).  Where the CPU lane takes experts its plan
#   comes from timings measured at load, so base vs base may differ there (GPU day 1): a FAIL with base != base is the
#   lane's noise, not the port's.
# ds4-ab: maya-ds4 greedy tokens (--n -Ds4N, the whole prompt in one window) after the -Ds4Ids prompt (comma-separated
#   ids, e.g. serve/ds41_encoding.py's prompt through the model's tokenizer.json), then the prompt's logits
#   (--dump-logits, the CLI's 16-token windows).  --profile-readonly and private copies of the pack's lt-picks.txt
#   (STRATA_DS4_LT_STORE) and lane-times.txt (DS4_LANE_CAL) leave the pack unwritten and give every run the same
#   prompt GEMMs; a first run of each build (not compared) fills those copies for this prompt's shapes and that build.
#   A model larger than the VRAM keeps the rest of its experts on the CPU lane, whose arithmetic is not the GPU's: then
#   (1) the VRAM expert cache moves experts as their background reads land, so which ones the GPU holds at a token is
#   timing's - the default -Ds4Args turn it off (--no-expert-cache: the experts stay where the load put them); and
#   (2) how many fit follows the memory free at load, so two loads can place a few experts differently and answer
#   differently, base against base too (v1.1.56, DeepSeek EXL3 2.9 bpw on the 8065S: 11,725 or 11,720 of 15,744 experts
#   in VRAM).  So runs are compared within groups of one placement (the log's "(N of M in VRAM)"), and the check needs a
#   base and a port run that share one; else it FAILs as inconclusive (more -Runs, or a model that fits the VRAM).
#   -NoDump skips the logits; -Tag names the files (<tag>.*) so that two packs can share one -Out.  A compared run that
#   exits non-zero FAILs its check by itself.
param(
    [Parameter(Position = 0, Mandatory = $true)][ValidateSet('flash-ab', 'ds4-ab', 'all')][string]$Cmd,
    [string]$FlashPack, [string]$AbTokens, [int]$Ctx = 2048, [int]$MaxNew = 256,
    [ValidateSet('off', 'default', 'both')][string]$Lane = 'off',
    [string]$Ds4Pack, [string]$Ds4Ids, [int]$Ds4N = 128, [switch]$NoDump, [string]$Tag = 'ds4',
    [string]$Ds4Args = '--max-context 131072 --tp 1 --threads on --reserve-mib 3072 --vram-experts 1.0 --mmap-experts auto --dspark on --no-expert-cache',
    [int]$Runs = 2, [string]$Base, [string]$Port, [string]$Out, [string]$LibDir, [switch]$Force)

$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
if (-not $Port) { $Port = @("$root\build-hip", "$root\build") | Where-Object { Test-Path "$_\maya.exe" } | Select-Object -First 1 }
if (-not $Base -or -not $Port) { Write-Output "-Base and -Port must name build folders with maya.exe"; exit 2 }
if (-not $Out) { $Out = Join-Path $env:TEMP ('glm53-gpu-' + (Get-Date -Format 'yyyyMMdd-HHmm')) }
if (-not $LibDir -and $env:HIP_PATH) { $LibDir = Join-Path $env:HIP_PATH 'bin' }
if ($LibDir) { $env:PATH = "$LibDir;$env:PATH" }
New-Item -ItemType Directory -Force $Out | Out-Null
$sum = Join-Path $Out 'summary.txt'
$script:fails = 0
function Note([string]$m) { $l = "$(Get-Date -Format 'HH:mm:ss') $m"; Add-Content -Path $sum -Value $l; Write-Output $l }
function Pass([string]$m) { Note "PASS $m" }
function Fail([string]$m) { $script:fails++; Note "FAIL $m" }
if (-not $Force -and (Get-NetTCPConnection -LocalPort 8080 -State Listen -ErrorAction SilentlyContinue)) {
    Write-Output "port 8080 is serving (the dashboard holds the GPU): stop it, then rerun (-Force skips this)"; exit 3
}
foreach ($e in @(Get-ChildItem Env: | Where-Object { $_.Name -match '^(STRATA_|MAYA_|DS4_)' })) {
    Note "note: $($e.Name)=$($e.Value) is set in this shell and reaches every run"
}

function Run([string]$exe, [string[]]$argv, [string]$stdout, [string]$stderr) {
    $t0 = Get-Date
    $p = Start-Process -FilePath $exe -ArgumentList $argv -RedirectStandardOutput $stdout -RedirectStandardError $stderr -NoNewWindow -PassThru
    $null = $p.Handle
    $p.WaitForExit()
    [pscustomobject]@{ rc = $p.ExitCode; min = '{0:N1}' -f ((Get-Date) - $t0).TotalMinutes }
}
function Hash([string]$f) { if ((Test-Path $f) -and (Get-Item $f).Length -gt 0) { (Get-FileHash $f).Hash } else { '' } }
function Last([string]$f, [string]$pat) {
    if (-not (Test-Path $f)) { return '' }
    $l = Select-String -Path $f -Pattern $pat | Select-Object -Last 1
    if ($l) { $l.Line.Trim() } else { '' }
}
# the runs that failed (exit code, ERR, no DONE line): their output file -> why; a check over one of them FAILs
$script:bad = @{}
# the first line where two text files part, for a FAIL's note
function FirstDiff([string]$a, [string]$b) {
    $la = @(Get-Content $a); $lb = @(Get-Content $b)
    for ($i = 0; $i -lt [Math]::Max($la.Count, $lb.Count); $i++) {
        $x = if ($i -lt $la.Count) { $la[$i] } else { '(end)' }
        $y = if ($i -lt $lb.Count) { $lb[$i] } else { '(end)' }
        if ($x -cne $y) {
            if ($x.Length -gt 200) { $x = $x.Substring(0, 200) + '...' }
            if ($y.Length -gt 200) { $y = $y.Substring(0, 200) + '...' }
            return " (line $($i + 1): '$x' against '$y')"
        }
    }
    ''
}
# every run of a list against the first: "PASS what" when all are equal, none is empty and none failed
function Agree([string]$what, [string[]]$files) {
    $failed = @($files | Where-Object { $script:bad.Contains($_) } | ForEach-Object { "$(Split-Path $_ -Leaf): $($script:bad[$_])" })
    if ($failed.Count -gt 0) { Fail "$what - a run failed: $($failed -join '; ')"; return }
    $h = @($files | ForEach-Object { Hash $_ })
    $names = ($files | ForEach-Object { Split-Path $_ -Leaf }) -join ', '
    if (($h | Where-Object { $_ -eq '' }).Count -gt 0) { Fail "$what missing or empty ($names)"; return }
    $odd = @(for ($i = 1; $i -lt $h.Count; $i++) {
        if ($h[$i] -ne $h[0]) { "$(Split-Path $files[$i] -Leaf)$(if ($files[$i] -like '*.tokens') { FirstDiff $files[0] $files[$i] })" }
    })
    if ($odd.Count -eq 0) { Pass "$what identical in $($h.Count) runs ($names)" }
    else { Fail "$what differ from $(Split-Path $files[0] -Leaf): $($odd -join ', ')" }
}
$order = @(for ($r = 1; $r -le $Runs; $r++) { "base.$r"; "port.$r" })
$bin = @{ base = $Base; port = $Port }

# a greedy run's output as flash-ab compares it: every line - the INFO / VRAM lines' expert tiers and memory layout
# (weights=, kv=, act=, drafter=, pool=, slots=, ...), the T lines, the DONE line's counts, finish and drafting counters
# (DONE produced prompt prompt-ms decode-ms finish accepted offered reuse hits lookups) - with only what follows the
# clock or the memory free at load masked: the DONE line's two wall-clock fields, and the VRAM lines' free= and window=
# (both read from the device's free memory, window= as its drop across the verify window's allocations: they move when
# anything else holds GPU memory - the desktop on a Windows APU - and with the size of the build's own GPU code on HIP:
# v1.1.56 against the port on the 8065S, free= 16 MiB and window= 140 KiB apart, every other field equal)
function Answer([string]$gen) {
    Get-Content $gen | ForEach-Object {
        if ($_ -match '^DONE ') {
            $d = $_ -split ' '
            foreach ($i in 3, 4) { if ($i -lt $d.Count) { $d[$i] = '-' } }
            $d -join ' '
        } elseif ($_ -match '^VRAM ') { $_ -replace ' (free|window)=\d+', ' $1=-' }
        else { $_ }
    }
}
# why a greedy run does not count ('' when it finished): its exit code, an ERR line, or no last line DONE whose token
# count is its T lines' (at least one)
function GenProblem([string]$gen, [int]$rc) {
    if ($rc -ne 0) { return "exit code $rc" }
    if (-not (Test-Path $gen)) { return 'no output' }
    $lines = @(Get-Content $gen)
    $err = @($lines | Where-Object { $_ -match '^ERR' })
    if ($err.Count -gt 0) { return $err[0] }
    $n = @($lines | Where-Object { $_ -match '^T ' }).Count
    if ($lines.Count -eq 0 -or -not ($lines[-1] -match '^DONE (\d+) ')) { return 'no DONE line at the end' }
    if ([int]$Matches[1] -ne $n -or $n -eq 0) { return "its DONE line counts $($Matches[1]) tokens, its T lines $n" }
    ''
}

function FlashAb {
    if (-not $FlashPack -or -not $AbTokens) { Write-Output "flash-ab needs -FlashPack and -AbTokens"; exit 2 }
    $ids = @(((Get-Content $AbTokens -Raw) -split '[,\s]+') | Where-Object { $_ -match '^\d+$' })
    if ($ids.Count -le $Ctx) { Write-Output "-AbTokens has $($ids.Count) ids: more than -Ctx ($Ctx) are needed"; exit 2 }
    $n = [Math]::Min($ids.Count, $Ctx + 256)
    ($ids[0..($n - 1)] -join ',') | Set-Content -NoNewline -Encoding ascii "$Out\flash.tokens"
    ($ids[0..($Ctx - 1)] -join ',') | Set-Content -NoNewline -Encoding ascii "$Out\flash.prompt"
    $lanes = if ($Lane -eq 'both') { @('default', 'off') } else { @($Lane) }
    foreach ($ln in $lanes) {
        if ($ln -eq 'off') { $env:STRATA_GLM_CPU_LANE = '0'; $env:STRATA_GLM_PREFILL_CPU = '0' }
        foreach ($run in $order) {
            $side = $run.Split('.')[0]
            $f = "$Out\flash.$ln.$run"
            Remove-Item "$f.score", "$f.tokens" -ErrorAction SilentlyContinue   # (an earlier run's, in a reused -Out)
            $env:STRATA_GLM_SCORE = "$Ctx"; $env:STRATA_GLM_SCORE_DUMP = "$f.score"
            $x = Run "$($bin[$side])\maya.exe" @('--glm-pack', $FlashPack, '--tokens-file', "$Out\flash.tokens") "$f.score.out" "$f.score.log"
            Remove-Item Env:STRATA_GLM_SCORE, Env:STRATA_GLM_SCORE_DUMP
            if ($x.rc -ne 0) { $script:bad["$f.score"] = "exit code $($x.rc)" }
            Note "flash $ln $run score: rc $($x.rc), $($x.min) min; $(Last "$f.score.out" '^SCORE')$(Last "$f.score.log" '^SCORE')"
            # the greedy run with the MTP draft length fixed (the cap, every round): the adaptive length (glm_mtp.cu
            # mtp_choose; the one-GPU MTP decode, on by default on the Halo APUs) follows measured round times, and
            # with it the DONE line's drafts and the decode's expert lookups - with it fixed they repeat and are compared
            $env:STRATA_GLM_MTP_ADAPT = '0'
            $x = Run "$($bin[$side])\maya.exe" @('--glm-pack', $FlashPack, '--tokens-file', "$Out\flash.prompt", '--greedy', '--max-new', "$MaxNew") "$f.gen" "$f.gen.log"
            Remove-Item Env:STRATA_GLM_MTP_ADAPT
            $why = GenProblem "$f.gen" $x.rc
            if ($why) { $script:bad["$f.tokens"] = $why }
            if ((Test-Path "$f.gen") -and (Get-Item "$f.gen").Length -gt 0) { Answer "$f.gen" | Set-Content -Encoding ascii "$f.tokens" }
            $done = Last "$f.gen" '^DONE '
            Note "flash $ln $run greedy: rc $($x.rc), $($x.min) min; $(@(Select-String -Path "$f.gen" -Pattern '^T ').Count) tokens; $done$(if ($why) { " - FAILED: $why" }); $(Last "$f.gen.log" '^glm stat: decode')"
        }
        Remove-Item Env:STRATA_GLM_CPU_LANE, Env:STRATA_GLM_PREFILL_CPU -ErrorAction SilentlyContinue
        Agree "flash-ab lane=$ln`: score dumps ($((Get-Item "$Out\flash.$ln.base.1.score" -ErrorAction SilentlyContinue).Length) B)" @($order | ForEach-Object { "$Out\flash.$ln.$_.score" })
        Agree "flash-ab lane=$ln`: $MaxNew greedy tokens with the memory layout and the DONE counts" @($order | ForEach-Object { "$Out\flash.$ln.$_.tokens" })
    }
}

# the load's expert placement, from a maya-ds4 log: "N/M" experts in VRAM ("-" when the log does not say)
function Placement([string]$log) {
    if ((Last $log 'in VRAM[)]') -match '\((\d+) of (\d+) in VRAM\)') { "$($Matches[1])/$($Matches[2])" } else { '-' }
}
# runs grouped by their placement: within a group every run must equal the group's first, and some group must hold a
# base run and a port run (else nothing compared the builds: FAIL, inconclusive)
function AgreeByPlacement([string]$what, [string[]]$runs, [string]$ext, [string]$logExt) {
    # (a failed run first: alone in its placement group it would not be compared at all)
    $failed = @($runs | Where-Object { $script:bad.Contains("$Out\$Tag.$_.$ext") } | ForEach-Object { "$_ ($($script:bad["$Out\$Tag.$_.$ext"]))" })
    if ($failed.Count -gt 0) { Fail "$what - a run failed: $($failed -join '; ')"; return }
    $groups = [ordered]@{}
    foreach ($r in $runs) {
        $pl = Placement "$Out\$Tag.$r.$logExt"
        if (-not $groups.Contains($pl)) { $groups[$pl] = @() }
        $groups[$pl] += $r
    }
    if ($groups.Count -gt 1) {
        Note ("note: $what - the expert placement differed between loads: " +
              (($groups.Keys | ForEach-Object { "$_ in VRAM: $($groups[$_] -join ' ')" }) -join '; '))
    }
    $both = $false
    foreach ($pl in $groups.Keys) {
        $g = @($groups[$pl])
        if ($g.Count -gt 1) { Agree "$what, placement $pl" @($g | ForEach-Object { "$Out\$Tag.$_.$ext" }) }
        if (@($g | Where-Object { $_ -like 'base.*' }).Count -gt 0 -and @($g | Where-Object { $_ -like 'port.*' }).Count -gt 0) { $both = $true }
    }
    if (-not $both) { Fail "$what`: no base run and port run shared an expert placement - inconclusive (more -Runs, or a model that fits the VRAM)" }
}

function Ds4Ab {
    if (-not $Ds4Pack -or -not $Ds4Ids) { Write-Output "ds4-ab needs -Ds4Pack and -Ds4Ids"; exit 2 }
    $lt = "$Out\$Tag.lt-picks.txt"
    if (-not (Test-Path $lt)) { if (Test-Path "$Ds4Pack\lt-picks.txt") { Copy-Item "$Ds4Pack\lt-picks.txt" $lt } else { New-Item -ItemType File $lt | Out-Null } }
    $cal = "$Out\$Tag.lane-times.txt"
    if (-not (Test-Path $cal) -and (Test-Path "$Ds4Pack\lane-times.txt")) { Copy-Item "$Ds4Pack\lane-times.txt" $cal }
    $env:STRATA_DS4_LT_STORE = $lt
    $env:DS4_LANE_CAL = $cal
    $common = @('--pack', $Ds4Pack) + @($Ds4Args -split '\s+' | Where-Object { $_ }) + @('--profile-readonly', '--ids', "@$Ds4Ids")
    foreach ($run in @('base.warm', 'port.warm') + $order) {
        $side = $run.Split('.')[0]
        $f = "$Out\$Tag.$run"
        $x = Run "$($bin[$side])\maya-ds4.exe" ($common + @('--window', '8192', '--n', "$Ds4N")) "$f.out" "$f.log"
        if ($x.rc -ne 0) { $script:bad["$f.tokens"] = "exit code $($x.rc)" }
        $toks = @([regex]::Matches((Get-Content "$f.out" -Raw) + '', '\d+') | ForEach-Object { $_.Value })
        ($toks -join ',') | Set-Content -NoNewline -Encoding ascii "$f.tokens"
        Note "$Tag $run greedy: rc $($x.rc), $($x.min) min; $($toks.Count) tokens; placement $(Placement "$f.log"); $(Last "$f.log" '^decode: \d+ tokens')"
    }
    if (-not $NoDump) {
        foreach ($run in $order) {
            $side = $run.Split('.')[0]
            $f = "$Out\$Tag.$run"
            Remove-Item "$f.f32" -ErrorAction SilentlyContinue   # (an earlier run's, in a reused -Out)
            $x = Run "$($bin[$side])\maya-ds4.exe" ($common + @('--n', '0', '--dump-logits', "$f.f32")) "$f.dump.out" "$f.dump.log"
            if ($x.rc -ne 0) { $script:bad["$f.f32"] = "exit code $($x.rc)" }
            Note "$Tag $run logits: rc $($x.rc), $($x.min) min; placement $(Placement "$f.dump.log"); $(Last "$f.dump.log" '^prompt: \d+ tokens')"
        }
    }
    Remove-Item Env:STRATA_DS4_LT_STORE, Env:DS4_LANE_CAL
    AgreeByPlacement "ds4-ab $Tag`: $Ds4N greedy tokens" $order 'tokens' 'log'
    if (-not $NoDump) {
        AgreeByPlacement "ds4-ab $Tag`: the prompt's logits ($((Get-Item "$Out\$Tag.base.1.f32" -ErrorAction SilentlyContinue).Length) B)" $order 'f32' 'dump.log'
    }
}

Note "glm53_gpu_checks $($Cmd): base $Base, port $Port, out $Out"
if ($Cmd -in 'flash-ab', 'all') { FlashAb }
if ($Cmd -in 'ds4-ab', 'all') { Ds4Ab }
Note ("summary: " + $(if ($script:fails -eq 0) { 'all PASS' } else { "$($script:fails) FAIL" }))
exit [int]($script:fails -gt 0)
