"""Build a thesis defense PPTX from final DMS evidence.

The environment does not require python-pptx. This script writes a minimal
PowerPoint Open XML package directly and includes speaker notes plus a
companion Markdown notes file.
"""

from __future__ import annotations

import html
import os
import shutil
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory


OUT_DIR = Path("/mnt/c/Users/tejac/Downloads/final_ppt_ref")
PPTX_OUT = OUT_DIR / "DMS_Final_Defense_Thesis_Findings.pptx"
NOTES_OUT = OUT_DIR / "DMS_Final_Defense_Thesis_Findings_Speaker_Notes.md"

ROOT = Path(".")
FIG_ROOT = ROOT / "analysis/figures/stage3/multivideo_final"
CORE = FIG_ROOT / "figures_core"
SEL = FIG_ROOT / "selected_video_deep_dive"

SLIDE_W = 12192000
SLIDE_H = 6858000
EMU_PER_IN = 914400

TITLE_COLOR = "0F172A"
ACCENT = "2563EB"
MUTED = "64748B"
BG = "F8FAFC"
WHITE = "FFFFFF"
DARK = "111827"
ORANGE = "EA580C"
GREEN = "16A34A"
RED = "DC2626"


@dataclass
class ImageSpec:
    path: Path
    x: int
    y: int
    w: int
    h: int


@dataclass
class TextBox:
    text: str
    x: int
    y: int
    w: int
    h: int
    size: int = 18
    color: str = DARK
    bold: bool = False
    bullets: bool = False
    fill: str | None = None
    border: str | None = None


@dataclass
class Slide:
    title: str
    subtitle: str = ""
    bullets: list[str] = field(default_factory=list)
    images: list[ImageSpec] = field(default_factory=list)
    boxes: list[TextBox] = field(default_factory=list)
    notes: str = ""
    footer: str = "Dynamic Model Selection for UAV Inspection"


def emu(inches: float) -> int:
    return int(inches * EMU_PER_IN)


def esc(s: str) -> str:
    return html.escape(str(s), quote=True)


def paragraph(text: str, size: int = 18, color: str = DARK, bold: bool = False, bullet: bool = False) -> str:
    ppr = f'<a:pPr><a:buChar char="•"/></a:pPr>' if bullet else "<a:pPr/>"
    b = ' b="1"' if bold else ""
    return (
        "<a:p>"
        f"{ppr}"
        f'<a:r><a:rPr lang="en-US" sz="{size * 100}"{b}><a:solidFill><a:srgbClr val="{color}"/></a:solidFill></a:rPr>'
        f"<a:t>{esc(text)}</a:t></a:r>"
        "</a:p>"
    )


def shape_text(shape_id: int, box: TextBox) -> str:
    if box.fill:
        fill = f'<a:solidFill><a:srgbClr val="{box.fill}"/></a:solidFill>'
        if box.border:
            fill += f'<a:ln w="12700"><a:solidFill><a:srgbClr val="{box.border}"/></a:solidFill></a:ln>'
        else:
            fill += '<a:ln><a:noFill/></a:ln>'
    else:
        fill = "<a:noFill/><a:ln><a:noFill/></a:ln>"
    lines = box.text.split("\n")
    paras = []
    for line in lines:
        if not line:
            paras.append("<a:p/>")
        else:
            paras.append(paragraph(line, box.size, box.color, box.bold, box.bullets))
    return f"""
    <p:sp>
      <p:nvSpPr><p:cNvPr id="{shape_id}" name="TextBox {shape_id}"/><p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr>
      <p:spPr><a:xfrm><a:off x="{box.x}" y="{box.y}"/><a:ext cx="{box.w}" cy="{box.h}"/></a:xfrm>{fill}</p:spPr>
      <p:txBody>
        <a:bodyPr wrap="square" lIns="91440" tIns="45720" rIns="91440" bIns="45720"/>
        <a:lstStyle/>
        {''.join(paras)}
      </p:txBody>
    </p:sp>
    """


def image_shape(shape_id: int, rel_id: str, img: ImageSpec) -> str:
    return f"""
    <p:pic>
      <p:nvPicPr><p:cNvPr id="{shape_id}" name="{esc(img.path.name)}"/><p:cNvPicPr/><p:nvPr/></p:nvPicPr>
      <p:blipFill><a:blip r:embed="{rel_id}"/><a:stretch><a:fillRect/></a:stretch></p:blipFill>
      <p:spPr><a:xfrm><a:off x="{img.x}" y="{img.y}"/><a:ext cx="{img.w}" cy="{img.h}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></p:spPr>
    </p:pic>
    """


def slide_xml(slide: Slide, idx: int, image_rels: list[tuple[str, str]]) -> str:
    shapes = []
    sid = 2
    shapes.append(shape_text(sid, TextBox(slide.title, emu(0.45), emu(0.28), emu(12.2), emu(0.55), 25, TITLE_COLOR, True)))
    sid += 1
    if slide.subtitle:
        shapes.append(shape_text(sid, TextBox(slide.subtitle, emu(0.48), emu(0.82), emu(12.0), emu(0.42), 12, MUTED)))
        sid += 1
    if slide.bullets:
        bullet_text = "\n".join(slide.bullets)
        shapes.append(shape_text(sid, TextBox(bullet_text, emu(0.65), emu(1.38), emu(5.2), emu(4.6), 16, DARK, False, True)))
        sid += 1
    for box in slide.boxes:
        shapes.append(shape_text(sid, box))
        sid += 1
    for img, (rid, _) in zip(slide.images, image_rels):
        shapes.append(image_shape(sid, rid, img))
        sid += 1
    shapes.append(shape_text(sid, TextBox(slide.footer, emu(0.45), emu(7.12), emu(8.0), emu(0.25), 8, MUTED)))
    sid += 1
    shapes.append(shape_text(sid, TextBox(str(idx), emu(12.45), emu(7.08), emu(0.35), emu(0.25), 8, MUTED)))

    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
       xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
       xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld>
    <p:bg><p:bgPr><a:solidFill><a:srgbClr val="{BG}"/></a:solidFill><a:effectLst/></p:bgPr></p:bg>
    <p:spTree>
      <p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>
      <p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{SLIDE_W}" cy="{SLIDE_H}"/><a:chOff x="0" y="0"/><a:chExt cx="{SLIDE_W}" cy="{SLIDE_H}"/></a:xfrm></p:grpSpPr>
      {''.join(shapes)}
    </p:spTree>
  </p:cSld>
  <p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:sld>"""


def rels_xml(rels: list[tuple[str, str, str]]) -> str:
    body = "\n".join(
        f'<Relationship Id="{rid}" Type="{typ}" Target="{esc(target)}"/>' for rid, typ, target in rels
    )
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
{body}
</Relationships>"""


def notes_xml(slide_idx: int, notes: str) -> str:
    paras = "".join(paragraph(line, 12, DARK) for line in notes.split("\n") if line.strip())
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:notes xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
         xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
         xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld>
    <p:spTree>
      <p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>
      <p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{SLIDE_W}" cy="{SLIDE_H}"/><a:chOff x="0" y="0"/><a:chExt cx="{SLIDE_W}" cy="{SLIDE_H}"/></a:xfrm></p:grpSpPr>
      <p:sp>
        <p:nvSpPr><p:cNvPr id="2" name="Notes Placeholder"/><p:cNvSpPr txBox="1"/><p:nvPr><p:ph type="body" idx="1"/></p:nvPr></p:nvSpPr>
        <p:spPr><a:xfrm><a:off x="685800" y="685800"/><a:ext cx="10972800" cy="5486400"/></a:xfrm><a:noFill/><a:ln><a:noFill/></a:ln></p:spPr>
        <p:txBody><a:bodyPr wrap="square"/><a:lstStyle/>{paras}</p:txBody>
      </p:sp>
    </p:spTree>
  </p:cSld>
  <p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:notes>"""


def add_card(slide: Slide, title: str, body: str, x: float, y: float, w: float, h: float, color: str = ACCENT) -> None:
    slide.boxes.append(TextBox(title, emu(x), emu(y), emu(w), emu(0.35), 16, color, True))
    slide.boxes.append(TextBox(body, emu(x), emu(y + 0.37), emu(w), emu(h - 0.37), 12, DARK, False, fill=WHITE, border="CBD5E1"))


def slides() -> list[Slide]:
    s: list[Slide] = []

    s1 = Slide(
        "Dynamic Model Selection for Real-Time UAV Inspection",
        "Final defense deck: runtime-quality tradeoffs, trigger validity, and cross-video evidence",
        notes="Open with the central idea: the system decides frame-by-frame whether a fast or accurate detector is worth running. State that the thesis is not a generic YOLO speed-up claim; it is a systems evaluation of when switching is useful.",
    )
    add_card(s1, "One-line thesis", "DMS is useful when accurate-model benefit, trigger intelligence, and runtime economy align.", 0.65, 1.55, 5.6, 1.35, ACCENT)
    add_card(s1, "Final evidence", "84 completed multi-video runs\n14 videos: 9 glass + 5 porcelain\n8 policies per run\nCPU and CUDA evaluated carefully", 6.6, 1.55, 5.5, 1.55, GREEN)
    add_card(s1, "Main caution", "Agreement is measured against the accurate detector reference, not dense human-labelled video mAP.", 3.2, 3.75, 6.7, 1.1, ORANGE)
    s.append(s1)

    s.append(Slide(
        "Talk Roadmap",
        "A guided proof rather than a data dump",
        bullets=[
            "Problem: fast detector vs accurate detector under onboard compute limits",
            "Method: DMS policies, deployed runtime accounting, random-baseline correction",
            "Evidence: 84-run cross-video benchmark plus selected-video deep dive",
            "Result: DMS is conditional; CPU and CUDA behave differently",
            "Transfer: Jetson and Anti-UAV as follow-up validation directions",
        ],
        notes="Tell the audience how to listen: each result answers one condition. First, is there a benefit? Second, can a trigger find it? Third, is it worth the runtime overhead?",
    ))

    s3 = Slide("Motivation: Why Dynamic Selection?", "UAV inspection needs speed and reliability at the same time", notes="Explain the tradeoff in simple systems terms. A small detector is fast but can miss objects. A large detector is more reliable but costs runtime. The thesis asks whether the system can decide which frame deserves the careful model.")
    add_card(s3, "Fast detector", "Low latency\nHigher FPS\nCan miss difficult frames", 0.7, 1.55, 3.4, 2.0, ACCENT)
    add_card(s3, "Accurate detector", "Higher agreement\nBetter evidence on difficult frames\nMore expensive on CPU", 4.9, 1.55, 3.4, 2.0, ORANGE)
    add_card(s3, "DMS policy", "Uses a lightweight signal\nSelects one detector per frame\nPays trigger overhead", 9.1, 1.55, 3.4, 2.0, GREEN)
    add_card(s3, "Thesis question", "When does dynamic switching create a defensible runtime-quality tradeoff for UAV inspection?", 2.0, 4.45, 9.4, 1.1, TITLE_COLOR)
    s.append(s3)

    s4 = Slide("Research Gap and Defense Logic", "DMS must satisfy three conditions", notes="This is the defense frame. If the accurate model does not help, switching is unnecessary. If the trigger is no better than random, switching is not intelligent. If the trigger overhead is too high, switching is not useful.")
    add_card(s4, "1. Accurate benefit", "Does the accurate model add useful evidence on this frame?\nMetric: benefit_positive / benefit_rate", 0.55, 1.45, 3.8, 2.1, ACCENT)
    add_card(s4, "2. Trigger validity", "Does the policy select benefit-positive frames better than random?\nMetric: informed_gain_over_random, precision/recall/F1", 4.75, 1.45, 3.8, 2.1, GREEN)
    add_card(s4, "3. Runtime economy", "Does the saved detector time exceed T_scene + T_ctrl overhead?\nMetric: policy-relevant T_total and FPS", 8.95, 1.45, 3.8, 2.1, ORANGE)
    add_card(s4, "Safe conclusion", "DMS is not a universal speed-up. It is useful when all three conditions align.", 2.15, 4.55, 8.8, 1.0, RED)
    s.append(s4)

    s5 = Slide("End-to-End DMS Architecture", "Frame → policy signal → detector selection → output", notes="Clarify deployment intent. In deployment, one policy is active and one detector is selected per frame. In offline evaluation, both models are run once so every policy can be evaluated fairly on the same frames.")
    add_card(s5, "Input frame", "UAV video frame", 0.55, 2.1, 2.0, 0.9, ACCENT)
    add_card(s5, "Policy signal", "Scene proxy features or fast-detector confidence state", 3.0, 1.85, 2.6, 1.4, GREEN)
    add_card(s5, "Switch decision", "Choose fast or accurate detector", 6.05, 1.95, 2.45, 1.2, ORANGE)
    add_card(s5, "Selected detector", "Only one detector is charged in deployed scene-policy accounting", 8.95, 1.75, 3.1, 1.55, ACCENT)
    add_card(s5, "Evaluation reference", "Chosen output is compared to accurate detector output for agreement metrics.", 2.0, 4.45, 9.2, 1.0, TITLE_COLOR)
    s.append(s5)

    s6 = Slide("Runtime Accounting: What Is T_total?", "Primary final tables use policy-relevant deployed runtime", notes="This slide is critical. State that previous raw logging computed all proxy features, but final thesis tables use policy-relevant T_scene. This avoids unfairly charging local_contrast_hyst for entropy, Laplacian, Tenengrad, and other logged features it does not use.")
    add_card(s6, "Scene-feature policies", "T_total = T_scene(policy-required) + T_ctrl + T_selected_detector", 0.65, 1.35, 5.9, 1.25, ACCENT)
    add_card(s6, "Confidence EMA", "T_total = T_fast + T_ctrl + I[accurate] × T_accurate", 6.9, 1.35, 5.1, 1.25, GREEN)
    add_card(s6, "T_scene by policy", "Entropy: H\nCombined: L + H\nLocal contrast hyst.: local_contrast\nNaive multi-proxy: L + H + Tenengrad + color entropy", 0.65, 3.05, 5.9, 1.95, ORANGE)
    add_card(s6, "Audit columns preserved", "t_total_policy_relevant_ms = thesis primary\nt_total_shared_proxy_ms = conservative full proxy/logging audit", 6.9, 3.05, 5.1, 1.95, TITLE_COLOR)
    s.append(s6)

    s7 = Slide("Experimental Coverage", "Final validation: 84 complete runs", notes="Use this slide to show scale and fairness. Every run has the same eight policies. CPU only has n_l pairs, so CPU-vs-CUDA conclusions use common n_l pair coverage.")
    add_card(s7, "Videos", "14 total videos\n9 glass\n5 porcelain", 0.7, 1.35, 2.7, 1.45, ACCENT)
    add_card(s7, "Models", "YOLOv8 and YOLO26\nn_s and n_l on CUDA\nn_l on CPU", 3.9, 1.35, 2.9, 1.45, GREEN)
    add_card(s7, "Policies", "8 policies per run\nfast_only, accurate_only, entropy, combined, conf_ema, multi_proxy, local_contrast_hyst", 7.25, 1.35, 4.9, 1.75, ORANGE)
    add_card(s7, "Comparison rule", "Strict platform comparisons use common n_l coverage only. This prevents mixing CUDA-only n_s runs with CPU n_l runs.", 1.6, 4.4, 10.0, 1.0, TITLE_COLOR)
    s.append(s7)

    s.append(Slide(
        "CPU Result: DMS Creates a Middle Ground",
        "Common n_l pair coverage; policy-relevant T_total",
        images=[ImageSpec(CORE / "fig_core_cpu_tradeoff.png", emu(0.75), emu(1.15), emu(7.25), emu(5.45))],
        boxes=[TextBox("How to read it\nLeft = faster\nHigher = closer to accurate detector\nDynamic policies sit between fast_only and accurate_only", emu(8.25), emu(1.7), emu(4.3), emu(2.2), 15, DARK, False, fill=WHITE, border="CBD5E1"),
               TextBox("Defense claim\nOn CPU, the fast/accurate latency gap creates room for dynamic switching.", emu(8.25), emu(4.35), emu(4.3), emu(1.15), 15, ACCENT, True, fill="EFF6FF", border="BFDBFE")],
        notes="Main CPU result. Dynamic policies improve agreement over fast_only while remaining faster than accurate_only. Do not say they beat accurate_only in quality; accurate_only is the upper reference.",
    ))

    s.append(Slide(
        "CUDA Result: Switching Is Not Automatically Faster",
        "Common n_l pair coverage; proxy overhead matters",
        images=[ImageSpec(CORE / "fig_core_cuda_tradeoff.png", emu(0.75), emu(1.15), emu(7.25), emu(5.45))],
        boxes=[TextBox("How to read it\nAccurate-only is already fast on CUDA.\nScene-proxy policies pay extra overhead.\nconf_ema remains relatively runtime-aware.", emu(8.25), emu(1.7), emu(4.3), emu(2.2), 15, DARK, False, fill=WHITE, border="CBD5E1"),
               TextBox("Defense claim\nDMS value is hardware-dependent, not universal.", emu(8.25), emu(4.35), emu(4.3), emu(1.15), 15, ORANGE, True, fill="FFF7ED", border="FED7AA")],
        notes="This is not a failure slide. It is the systems insight. If the accurate model is already close to the fast model in latency, adding a proxy can hurt runtime.",
    ))

    s.append(Slide(
        "Latency and FPS Benchmark",
        "Reader-friendly runtime summary using corrected T_total",
        images=[ImageSpec(CORE / "fig_core_latency_fps.png", emu(0.65), emu(1.05), emu(8.2), emu(5.65))],
        boxes=[TextBox("Key numbers, CPU n_l\nfast_only: 86.15 ms / 14.36 FPS\naccurate_only: 289.14 ms / 3.75 FPS\nconf_ema: 191.23 ms / 5.83 FPS\nlocal_contrast_hyst: 207.84 ms / 5.42 FPS", emu(9.05), emu(1.55), emu(3.35), emu(2.45), 13, DARK, False, fill=WHITE, border="CBD5E1"),
               TextBox("Use this slide when asked: 'Is it real-time?'", emu(9.05), emu(4.45), emu(3.35), emu(0.9), 14, ACCENT, True, fill="EFF6FF", border="BFDBFE")],
        notes="Explain latency and FPS together. On CPU, dynamic policies are slower than fast_only but substantially faster than accurate_only while gaining agreement. On CUDA, accurate_only remains very fast.",
    ))

    s.append(Slide(
        "Trigger Intelligence: Beating Same-Usage Random",
        "Raw agreement alone is not enough",
        images=[ImageSpec(CORE / "fig_core_informed_gain.png", emu(0.75), emu(1.15), emu(7.6), emu(5.2))],
        boxes=[TextBox("Why this matters\nA policy can look strong just by choosing accurate often.\nInformed gain compares against random switching at the same accurate usage.", emu(8.6), emu(1.6), emu(3.8), emu(2.2), 15, DARK, False, fill=WHITE, border="CBD5E1"),
               TextBox("Positive gain = actual trigger signal.\nNegative gain = worse than random at same usage.", emu(8.6), emu(4.3), emu(3.8), emu(1.1), 15, GREEN, True, fill="F0FDF4", border="BBF7D0")],
        notes="This is one of the most defensible research slides. It prevents overclaiming raw IoU agreement. Point out that multi_proxy is a weak naive baseline, not proof against all multi-feature policies.",
    ))

    s.append(Slide(
        "Trigger Precision, Recall, and F1",
        "Alignment with frames where the accurate model actually helps",
        images=[ImageSpec(CORE / "fig_core_trigger_precision_recall_f1.png", emu(0.65), emu(1.05), emu(8.3), emu(5.45))],
        boxes=[TextBox("Definitions\nPrecision: chosen accurate frames that were benefit-positive\nRecall: benefit-positive frames found by the policy\nF1: balance of both", emu(9.1), emu(1.6), emu(3.3), emu(2.2), 14, DARK, False, fill=WHITE, border="CBD5E1"),
               TextBox("Modest F1 is expected: benefit-positive frames are sparse and hard.", emu(9.1), emu(4.25), emu(3.3), emu(1.0), 14, ORANGE, True, fill="FFF7ED", border="FED7AA")],
        notes="Use this slide to explain that trigger quality is not just switching often. A good trigger should align with benefit-positive frames, but the task is inherently difficult.",
    ))

    s.append(Slide(
        "Selected-Video Deep Dive",
        "Four videos chosen for explainable CPU n_l behavior",
        images=[ImageSpec(SEL / "figures_cross_selected/selected_videos_runtime_quality_overview.png", emu(0.65), emu(1.1), emu(7.4), emu(5.15))],
        boxes=[TextBox("Selected videos\nGlass: glass_ins, 161_YUN_0001_96\nPorcelain: UAV_porcelain, porcelain_maybe\nModels: YOLOv8 n_l and YOLO26 n_l\nDevice: CPU", emu(8.35), emu(1.45), emu(4.0), emu(2.25), 14, DARK, False, fill=WHITE, border="CBD5E1"),
               TextBox("Purpose\nMake the 84-run evidence explainable using representative videos and per-frame timelines.", emu(8.35), emu(4.25), emu(4.0), emu(1.15), 14, ACCENT, True, fill="EFF6FF", border="BFDBFE")],
        notes="Transition from aggregate evidence to selected examples. State that these are not cherry-picked as final proof; they are explanatory case studies backed by the full 84-run benchmark.",
    ))

    s.append(Slide(
        "Selected Videos: Trigger Behavior Varies",
        "Informed gain over random across videos and model families",
        images=[ImageSpec(SEL / "figures_cross_selected/selected_videos_informed_gain_heatmap.png", emu(0.75), emu(1.05), emu(8.1), emu(5.45))],
        boxes=[TextBox("How to read it\nRows are video/model configurations.\nColumns are dynamic policies.\nPositive values indicate better-than-random selection.", emu(9.05), emu(1.55), emu(3.35), emu(2.2), 14, DARK, False, fill=WHITE, border="CBD5E1"),
               TextBox("Message\nNo single policy is universally best; deployment should be domain- and hardware-aware.", emu(9.05), emu(4.25), emu(3.35), emu(1.1), 14, GREEN, True, fill="F0FDF4", border="BBF7D0")],
        notes="Use this to show variation rather than hide it. It supports a robust thesis because the conclusion is conditional and measured.",
    ))

    s.append(Slide(
        "Example: Runtime Components Per Policy",
        "Defending T_scene, T_ctrl, and inference timing",
        images=[ImageSpec(SEL / "figures_per_run/glass_glass_ins_yolov8_n_l_cpu_runtime_components.png", emu(0.55), emu(1.0), emu(8.4), emu(5.55))],
        boxes=[TextBox("This figure answers:\nWhat exactly is each policy paying for?\nHow much is proxy cost?\nHow much is detector inference?\nIs controller time meaningful?", emu(9.15), emu(1.55), emu(3.25), emu(2.25), 14, DARK, False, fill=WHITE, border="CBD5E1"),
               TextBox("Benchmarking rule\nDo not charge a policy for unused image features.", emu(9.15), emu(4.35), emu(3.25), emu(0.95), 14, RED, True, fill="FEF2F2", border="FECACA")],
        notes="This slide directly addresses runtime-accounting questions. State that T_scene is policy-required feature cost, not the cost of every feature logged for analysis.",
    ))

    s.append(Slide(
        "Example: Benefit-Positive Frames and Switching",
        "Do policy switches occur where the accurate model helps?",
        images=[ImageSpec(SEL / "figures_per_run/porcelain_UAV_porcelain_yolov8_n_l_cpu_benefit_and_switching_timeline.png", emu(0.55), emu(1.0), emu(8.5), emu(5.55))],
        boxes=[TextBox("Black curve\nRolling benefit_positive rate: where accurate model adds useful evidence.", emu(9.2), emu(1.55), emu(3.2), emu(1.4), 14, DARK, False, fill=WHITE, border="CBD5E1"),
               TextBox("Colored curves\nRolling accurate usage by selected policies.\nAlignment means the trigger follows difficult/useful regions.", emu(9.2), emu(3.35), emu(3.2), emu(1.7), 14, DARK, False, fill=WHITE, border="CBD5E1")],
        notes="This is a normal-tech-audience explanation. The black curve is where using accurate is useful. The policy curves show how often policies choose accurate around those segments.",
    ))

    s.append(Slide(
        "Feature Validity: Structural-Texture Signals",
        "Image features predict benefit-positive frames, but not causally",
        images=[ImageSpec(CORE / "fig_core_structural_feature_counts.png", emu(0.65), emu(1.05), emu(7.5), emu(5.3))],
        boxes=[TextBox("Safe claim\nStructural-texture complexity is predictively associated with accurate-model benefit.", emu(8.45), emu(1.45), emu(3.85), emu(1.1), 15, GREEN, True, fill="F0FDF4", border="BBF7D0"),
               TextBox("Features\nlocal_contrast\nedge_density\nlaplacian\nTenengrad", emu(8.45), emu(2.95), emu(3.85), emu(1.45), 15, DARK, False, fill=WHITE, border="CBD5E1"),
               TextBox("Caution\nFeature ranking varies by video, domain, model pair, device, and lighting.", emu(8.45), emu(4.85), emu(3.85), emu(0.9), 13, ORANGE, True, fill="FFF7ED", border="FED7AA")],
        notes="Do not say local contrast is universally best. The final claim is feature-family level: structural-texture features are useful predictors under the evaluated distribution.",
    ))

    s.append(Slide(
        "Example: Feature PR Curves",
        "Which image features separate benefit-positive frames?",
        images=[ImageSpec(SEL / "figures_per_run/glass_glass_ins_yolov8_n_l_cpu_feature_pr_curves.png", emu(0.75), emu(1.1), emu(7.4), emu(5.25))],
        boxes=[TextBox("Precision-recall curve\nShows how well a feature ranks frames where the accurate model helps.", emu(8.45), emu(1.65), emu(3.8), emu(1.45), 14, DARK, False, fill=WHITE, border="CBD5E1"),
               TextBox("AUPRC lift\nUseful under class imbalance because benefit-positive frames may be sparse.", emu(8.45), emu(3.55), emu(3.8), emu(1.2), 14, ACCENT, True, fill="EFF6FF", border="BFDBFE")],
        notes="Explain PR curves instead of only AUROC. AUROC can look good even under imbalance, so AUPRC lift over the base rate is important.",
    ))

    s17 = Slide("Thresholds and Ablations", "Fixed operating points, not global optima", notes="Be precise: detector thresholds are fixed and uniform. Policy thresholds are calibrated operating points. Ablations test sensitivity to proxy size and update cadence; they do not prove global optimality.")
    add_card(s17, "Detector thresholds", "conf_floor = 0.25\niou_threshold = 0.5\nHeld fixed across runs", 0.7, 1.35, 3.4, 1.55, ACCENT)
    add_card(s17, "Policy thresholds", "Fixed/default or exploratory\nNo per-video tuning in final validation", 4.85, 1.35, 3.4, 1.55, GREEN)
    add_card(s17, "Ablations", "Proxy size\nFrame skip\nProxy stride / caching\nThreshold sensitivity", 9.0, 1.35, 3.4, 1.55, ORANGE)
    add_card(s17, "Defense wording", "The thesis claims measured behavior under fixed operating points, not mathematically optimal thresholds.", 2.0, 4.3, 9.4, 1.0, TITLE_COLOR)
    s.append(s17)

    s18 = Slide("Limitations and Scope Control", "The work is stronger because the claims are bounded", notes="Use this slide proactively. It shows that you understand the evidence boundary. This is important for a strong defense.")
    add_card(s18, "Reference label", "Accurate-reference agreement, not dense human video mAP.", 0.65, 1.3, 3.6, 1.2, RED)
    add_card(s18, "Trigger validity", "Predictive association, not causality.", 4.85, 1.3, 3.25, 1.2, ORANGE)
    add_card(s18, "Hardware", "CUDA behavior is RTX-specific; Jetson thermal/power evidence remains follow-up.", 8.65, 1.3, 3.6, 1.2, ACCENT)
    add_card(s18, "Features", "No single image feature is universal across lighting and domains.", 0.65, 3.25, 3.6, 1.2, GREEN)
    add_card(s18, "Policies", "multi_proxy is a naive fusion baseline, not a proof against multi-feature triggers.", 4.85, 3.25, 3.25, 1.2, TITLE_COLOR)
    add_card(s18, "Runtime", "Final ranking uses policy-relevant T_total; shared-proxy timing is retained as audit.", 8.65, 3.25, 3.6, 1.2, ACCENT)
    s.append(s18)

    s19 = Slide("Key Contributions", "What the thesis adds", notes="Summarize concretely. Avoid vague claims like 'improved YOLO'. The contributions are about systems evaluation, accounting, and trigger validity.")
    s19.bullets = [
        "Implemented a DMS framework over fast/accurate YOLO detector pairs.",
        "Defined deployed runtime accounting with T_scene, T_ctrl, and detector inference.",
        "Introduced benefit_positive as an explainable frame-level opportunity label.",
        "Evaluated policies using accurate-reference agreement and random-baseline correction.",
        "Validated across 84 complete glass/porcelain UAV inspection runs.",
        "Showed platform dependence: CPU can benefit; CUDA can be overhead-limited.",
    ]
    s.append(s19)

    s20 = Slide("Conclusion and Next Steps", "DMS is useful when three conditions align", notes="Close with the three-condition conclusion. Mention Jetson and Anti-UAV as extensions, not as necessary to prove the current core result.")
    add_card(s20, "Final takeaway", "DMS is a conditional runtime-quality tradeoff, not a universal speedup.", 0.8, 1.3, 5.5, 1.2, ACCENT)
    add_card(s20, "When it works", "1. Accurate model adds benefit\n2. Trigger beats same-usage random\n3. Runtime economy is favorable", 6.8, 1.3, 5.2, 1.8, GREEN)
    add_card(s20, "Next steps", "Jetson deployment with thermal/power logging\nAnti-UAV transfer validation\nAnnotated demo videos from selected segments\nCalibrated multi-feature triggers", 1.25, 4.05, 10.5, 1.6, ORANGE)
    s.append(s20)

    return s


def write_package(slides_list: list[Slide]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory() as td:
        root = Path(td)
        (root / "_rels").mkdir()
        (root / "docProps").mkdir()
        (root / "ppt/_rels").mkdir(parents=True)
        (root / "ppt/slides/_rels").mkdir(parents=True)
        (root / "ppt/notesSlides/_rels").mkdir(parents=True)
        (root / "ppt/slideLayouts/_rels").mkdir(parents=True)
        (root / "ppt/slideMasters/_rels").mkdir(parents=True)
        (root / "ppt/theme").mkdir(parents=True)
        (root / "ppt/media").mkdir(parents=True)

        content_overrides = [
            '<Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>',
            '<Override PartName="/ppt/slideMasters/slideMaster1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideMaster+xml"/>',
            '<Override PartName="/ppt/slideLayouts/slideLayout1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideLayout+xml"/>',
            '<Override PartName="/ppt/theme/theme1.xml" ContentType="application/vnd.openxmlformats-officedocument.theme+xml"/>',
            '<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>',
            '<Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>',
        ]

        media_idx = 1
        slide_rel_entries = []
        for idx, slide in enumerate(slides_list, 1):
            image_rels: list[tuple[str, str]] = []
            rels = [("rId1", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout", "../slideLayouts/slideLayout1.xml")]
            ridn = 2
            for img in slide.images:
                if not img.path.exists():
                    raise FileNotFoundError(img.path)
                ext = img.path.suffix.lower().lstrip(".") or "png"
                media_name = f"image{media_idx}.{ext}"
                shutil.copyfile(img.path, root / "ppt/media" / media_name)
                rid = f"rId{ridn}"
                rels.append((rid, "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image", f"../media/{media_name}"))
                image_rels.append((rid, media_name))
                media_idx += 1
                ridn += 1
            rels.append((f"rId{ridn}", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/notesSlide", f"../notesSlides/notesSlide{idx}.xml"))

            (root / f"ppt/slides/slide{idx}.xml").write_text(slide_xml(slide, idx, image_rels), encoding="utf-8")
            (root / f"ppt/slides/_rels/slide{idx}.xml.rels").write_text(rels_xml(rels), encoding="utf-8")
            (root / f"ppt/notesSlides/notesSlide{idx}.xml").write_text(notes_xml(idx, slide.notes), encoding="utf-8")
            (root / f"ppt/notesSlides/_rels/notesSlide{idx}.xml.rels").write_text(
                rels_xml([("rId1", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide", f"../slides/slide{idx}.xml")]),
                encoding="utf-8",
            )
            content_overrides.append(f'<Override PartName="/ppt/slides/slide{idx}.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>')
            content_overrides.append(f'<Override PartName="/ppt/notesSlides/notesSlide{idx}.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.notesSlide+xml"/>')
            slide_rel_entries.append((idx, f"rId{idx}"))

        (root / "[Content_Types].xml").write_text(f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Default Extension="png" ContentType="image/png"/>
  <Default Extension="jpg" ContentType="image/jpeg"/>
  <Default Extension="jpeg" ContentType="image/jpeg"/>
  {''.join(content_overrides)}
</Types>""", encoding="utf-8")

        (root / "_rels/.rels").write_text(rels_xml([
            ("rId1", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument", "ppt/presentation.xml"),
            ("rId2", "http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties", "docProps/core.xml"),
            ("rId3", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties", "docProps/app.xml"),
        ]), encoding="utf-8")

        slide_ids = "\n".join(f'<p:sldId id="{255 + i}" r:id="{rid}"/>' for i, rid in slide_rel_entries)
        (root / "ppt/presentation.xml").write_text(f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:presentation xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
                xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
                xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:sldMasterIdLst><p:sldMasterId id="2147483648" r:id="rId{len(slides_list)+1}"/></p:sldMasterIdLst>
  <p:sldIdLst>{slide_ids}</p:sldIdLst>
  <p:sldSz cx="{SLIDE_W}" cy="{SLIDE_H}" type="wide"/>
  <p:notesSz cx="6858000" cy="9144000"/>
</p:presentation>""", encoding="utf-8")

        pres_rels = [(f"rId{i}", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide", f"slides/slide{i}.xml") for i in range(1, len(slides_list) + 1)]
        pres_rels.append((f"rId{len(slides_list)+1}", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster", "slideMasters/slideMaster1.xml"))
        (root / "ppt/_rels/presentation.xml.rels").write_text(rels_xml(pres_rels), encoding="utf-8")

        (root / "ppt/slideMasters/slideMaster1.xml").write_text(f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sldMaster xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{SLIDE_W}" cy="{SLIDE_H}"/><a:chOff x="0" y="0"/><a:chExt cx="{SLIDE_W}" cy="{SLIDE_H}"/></a:xfrm></p:grpSpPr></p:spTree></p:cSld>
  <p:clrMap bg1="lt1" tx1="dk1" bg2="lt2" tx2="dk2" accent1="accent1" accent2="accent2" accent3="accent3" accent4="accent4" accent5="accent5" accent6="accent6" hlink="hlink" folHlink="folHlink"/>
  <p:sldLayoutIdLst><p:sldLayoutId id="2147483649" r:id="rId1"/></p:sldLayoutIdLst>
  <p:txStyles><p:titleStyle/><p:bodyStyle/><p:otherStyle/></p:txStyles>
</p:sldMaster>""", encoding="utf-8")
        (root / "ppt/slideMasters/_rels/slideMaster1.xml.rels").write_text(rels_xml([
            ("rId1", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout", "../slideLayouts/slideLayout1.xml"),
            ("rId2", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme", "../theme/theme1.xml"),
        ]), encoding="utf-8")

        (root / "ppt/slideLayouts/slideLayout1.xml").write_text(f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sldLayout xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" type="blank" preserve="1">
  <p:cSld name="Blank"><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{SLIDE_W}" cy="{SLIDE_H}"/><a:chOff x="0" y="0"/><a:chExt cx="{SLIDE_W}" cy="{SLIDE_H}"/></a:xfrm></p:grpSpPr></p:spTree></p:cSld>
  <p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:sldLayout>""", encoding="utf-8")
        (root / "ppt/slideLayouts/_rels/slideLayout1.xml.rels").write_text(rels_xml([
            ("rId1", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster", "../slideMasters/slideMaster1.xml"),
        ]), encoding="utf-8")

        (root / "ppt/theme/theme1.xml").write_text(f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<a:theme xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" name="DMS Thesis Theme">
  <a:themeElements>
    <a:clrScheme name="DMS"><a:dk1><a:srgbClr val="111827"/></a:dk1><a:lt1><a:srgbClr val="FFFFFF"/></a:lt1><a:dk2><a:srgbClr val="0F172A"/></a:dk2><a:lt2><a:srgbClr val="F8FAFC"/></a:lt2><a:accent1><a:srgbClr val="2563EB"/></a:accent1><a:accent2><a:srgbClr val="EA580C"/></a:accent2><a:accent3><a:srgbClr val="16A34A"/></a:accent3><a:accent4><a:srgbClr val="DC2626"/></a:accent4><a:accent5><a:srgbClr val="7C3AED"/></a:accent5><a:accent6><a:srgbClr val="0891B2"/></a:accent6><a:hlink><a:srgbClr val="2563EB"/></a:hlink><a:folHlink><a:srgbClr val="7C3AED"/></a:folHlink></a:clrScheme>
    <a:fontScheme name="Aptos"><a:majorFont><a:latin typeface="Aptos Display"/></a:majorFont><a:minorFont><a:latin typeface="Aptos"/></a:minorFont></a:fontScheme>
    <a:fmtScheme name="DMS"><a:fillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:fillStyleLst><a:lnStyleLst><a:ln w="9525"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln></a:lnStyleLst><a:effectStyleLst><a:effectStyle><a:effectLst/></a:effectStyle></a:effectStyleLst><a:bgFillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:bgFillStyleLst></a:fmtScheme>
  </a:themeElements>
</a:theme>""", encoding="utf-8")

        (root / "docProps/core.xml").write_text("""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"><dc:title>DMS Final Defense Thesis Findings</dc:title><dc:creator>Teja</dc:creator><cp:lastModifiedBy>Codex</cp:lastModifiedBy></cp:coreProperties>""", encoding="utf-8")
        (root / "docProps/app.xml").write_text(f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes"><Application>Codex</Application><PresentationFormat>On-screen Show (16:9)</PresentationFormat><Slides>{len(slides_list)}</Slides></Properties>""", encoding="utf-8")

        with zipfile.ZipFile(PPTX_OUT, "w", zipfile.ZIP_DEFLATED) as z:
            for path in root.rglob("*"):
                if path.is_file():
                    z.write(path, path.relative_to(root).as_posix())


def write_notes(slides_list: list[Slide]) -> None:
    lines = ["# DMS Final Defense Thesis Findings - Speaker Notes", ""]
    for i, slide in enumerate(slides_list, 1):
        lines.extend([f"## Slide {i}: {slide.title}", "", slide.notes.strip(), ""])
    NOTES_OUT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    slide_list = slides()
    write_package(slide_list)
    write_notes(slide_list)
    print(PPTX_OUT)
    print(NOTES_OUT)
    print(f"slides={len(slide_list)}")


if __name__ == "__main__":
    main()
