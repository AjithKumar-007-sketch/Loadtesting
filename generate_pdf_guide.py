#!/usr/bin/env python3
"""
Generates the LiveKit Voice AI Testing Framework Architecture & Executive Guide PDF.
"""

from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, KeepTogether
)

pdf_path = "LiveKit_Voice_AI_Testing_Framework_Executive_Guide.pdf"
doc = SimpleDocTemplate(
    pdf_path,
    pagesize=letter,
    leftMargin=36,
    rightMargin=36,
    topMargin=36,
    bottomMargin=36
)

styles = getSampleStyleSheet()

primary_color = colors.HexColor("#1A365D")    # Deep Navy
secondary_color = colors.HexColor("#2B6CB0")  # Steel Blue
text_dark = colors.HexColor("#2D3748")        # Charcoal
bg_light = colors.HexColor("#F7FAFC")         # Soft Off-White
border_color = colors.HexColor("#CBD5E0")     # Gray border

title_style = ParagraphStyle(
    "DocTitle",
    parent=styles["Normal"],
    fontName="Helvetica-Bold",
    fontSize=20,
    leading=24,
    textColor=primary_color,
    spaceAfter=4
)

subtitle_style = ParagraphStyle(
    "DocSubTitle",
    parent=styles["Normal"],
    fontName="Helvetica",
    fontSize=10,
    leading=14,
    textColor=colors.HexColor("#4A5568"),
    spaceAfter=10
)

h1_style = ParagraphStyle(
    "H1",
    parent=styles["Normal"],
    fontName="Helvetica-Bold",
    fontSize=12,
    leading=16,
    textColor=primary_color,
    spaceBefore=10,
    spaceAfter=4
)

body_style = ParagraphStyle(
    "Body",
    parent=styles["Normal"],
    fontName="Helvetica",
    fontSize=8.5,
    leading=12,
    textColor=text_dark,
    spaceAfter=4
)

bullet_style = ParagraphStyle(
    "Bullet",
    parent=body_style,
    leftIndent=12,
    firstLineIndent=-8,
    spaceAfter=3
)

table_header_style = ParagraphStyle(
    "TH",
    parent=styles["Normal"],
    fontName="Helvetica-Bold",
    fontSize=8,
    leading=10,
    textColor=colors.white
)

table_body_style = ParagraphStyle(
    "TB",
    parent=styles["Normal"],
    fontName="Helvetica",
    fontSize=7.8,
    leading=10,
    textColor=text_dark
)

table_code_style = ParagraphStyle(
    "TCode",
    parent=styles["Normal"],
    fontName="Courier-Bold",
    fontSize=7.8,
    leading=10,
    textColor=secondary_color
)

story = []

# Title Banner
story.append(Paragraph("LiveKit Voice AI Testing Framework", title_style))
story.append(Paragraph("Architecture, Latency Measurement Methodology &amp; Manager Executive Guide", subtitle_style))
story.append(HRFlowable(width="100%", thickness=1.5, color=primary_color, spaceAfter=8))

# Section 1: Executive Summary
story.append(Paragraph("1. Executive Summary", h1_style))
story.append(Paragraph(
    "In Conversational Voice AI, <b>latency directly drives customer satisfaction (CSAT)</b>. "
    "If conversational delays exceed 1.5 to 2.0 seconds, human callers perceive the AI agent as robotic, hesitant, or unresponsive. "
    "This framework provides an automated, scientifically verifiable benchmarking system that connects as a real WebRTC participant, "
    "measures sub-second latency at every pipeline stage, and verifies conversational integrity under multi-turn load.",
    body_style
))

# Section 2: Room Joining Architecture
story.append(Paragraph("2. Room Joining Architecture (How Caller &amp; Agent Connect)", h1_style))
story.append(Paragraph(
    "The framework connects directly to LiveKit as a <b>genuine WebRTC participant</b>, identical to a human caller in a web browser:",
    body_style
))

room_flow_data = [
    [Paragraph("Step", table_header_style), Paragraph("Component", table_header_style), Paragraph("Protocol", table_header_style), Paragraph("Technical Execution", table_header_style)],
    [Paragraph("<b>1</b>", table_body_style), Paragraph("Call Allocation", table_body_style), Paragraph("HTTP REST", table_code_style), Paragraph("Script sends POST /v1/agents/{id}/web-calls. The wrapper creates a room and returns an authenticated LiveKit JWT token.", table_body_style)],
    [Paragraph("<b>2</b>", table_body_style), Paragraph("Signaling &amp; Handshake", table_body_style), Paragraph("WebSockets", table_code_style), Paragraph("Caller joins room via room.connect(url, token), completing WebRTC signaling, ICE candidate gathering, and DTLS security.", table_body_style)],
    [Paragraph("<b>3</b>", table_body_style), Paragraph("Agent Worker Dispatch", table_body_style), Paragraph("LiveKit Dispatch", table_code_style), Paragraph("LiveKit server matches the room to an active Voice Agent worker process.", table_body_style)],
    [Paragraph("<b>4</b>", table_body_style), Paragraph("Microphone Publishing", table_body_style), Paragraph("WebRTC Audio", table_code_style), Paragraph("Caller creates LocalAudioTrack streaming 16 kHz mono audio frames into the room as the caller microphone.", table_body_style)],
    [Paragraph("<b>5</b>", table_body_style), Paragraph("Agent Voice Ingestion", table_body_style), Paragraph("WebRTC Audio", table_code_style), Paragraph("Caller subscribes to the agent's published audio track, recording raw voice samples to detect speech vs silence.", table_body_style)],
]

t_flow = Table(room_flow_data, colWidths=[24, 110, 75, 331])
t_flow.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), primary_color),
    ("GRID", (0, 0), (-1, -1), 0.5, border_color),
    ("TOPPADDING", (0, 0), (-1, -1), 3),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, bg_light]),
]))
story.append(t_flow)
story.append(Spacer(1, 6))

# Section 3: Latency Measurement Methodology
story.append(Paragraph("3. Latency Measurement Methodology (Mathematical Ground Truth)", h1_style))
story.append(Paragraph(
    "Latency is not guessed from API logs. It is computed at the <b>physical audio sample level</b> using "
    "Root-Mean-Square (RMS) audio energy detection (SPEECH_RMS &gt; 400 at 16 kHz) tied to monotonic system clocks:",
    body_style
))

latency_data = [
    [Paragraph("Metric Key", table_header_style), Paragraph("Pipeline Stage", table_header_style), Paragraph("Definition &amp; Formula", table_header_style), Paragraph("Business Impact", table_header_style)],
    [Paragraph("web_call_ms", table_code_style), Paragraph("REST API Gateway", table_body_style), Paragraph("Time to POST /web-calls and receive room token.", table_body_style), Paragraph("Initial call dial latency.", table_body_style)],
    [Paragraph("room_join_ms", table_code_style), Paragraph("WebRTC Signaling", table_body_style), Paragraph("Time elapsed from room.connect() until WebRTC peer is established.", table_body_style), Paragraph("Connection establishment time.", table_body_style)],
    [Paragraph("call_setup_ms", table_code_style), Paragraph("Call Onset (Greeting)", table_body_style), Paragraph("Total milliseconds from clicking Call until the first agent word is heard.", table_body_style), Paragraph("Caller time-to-first-sound.", table_body_style)],
    [Paragraph("agent_response_ms", table_code_style), Paragraph("Conversational Turn Latency", table_body_style), Paragraph("Time elapsed from caller finishing an utterance until agent voice starts.", table_body_style), Paragraph("<b>Critical SLA:</b> Determines conversational speed &amp; natural flow.", table_body_style)],
    [Paragraph("reply_duration_ms", table_code_style), Paragraph("Agent Speech Duration", table_body_style), Paragraph("Total duration of the agent's utterance until end-of-speech silence.", table_body_style), Paragraph("Agent verbosity and efficiency.", table_body_style)],
]

t_lat = Table(latency_data, colWidths=[95, 105, 200, 140])
t_lat.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), secondary_color),
    ("GRID", (0, 0), (-1, -1), 0.5, border_color),
    ("TOPPADDING", (0, 0), (-1, -1), 3),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, bg_light]),
]))
story.append(t_lat)
story.append(Spacer(1, 6))

# Section 4: Testing Modes
story.append(Paragraph("4. Testing Paradigms Supported", h1_style))
story.append(Paragraph("• <b>Single-Turn Latency Test:</b> Quick regression test verifying room join and agent greeting turnaround times.", bullet_style))
story.append(Paragraph("• <b>Deterministic Multi-Turn Scenarios:</b> Executes standardized scripted patient flows (e.g. Appointment Booking) to measure turn-by-turn latency.", bullet_style))
story.append(Paragraph("• <b>Dynamic Multimodal AI Caller (Gemini / OpenAI):</b> Uses native audio AI to listen directly to the agent and reply dynamically according to individual patient profiles (e.g. Anna Beckett, DOB 1973-10-04), adapting automatically to any question order.", bullet_style))
story.append(Paragraph("• <b>Concurrency &amp; Stress Benchmarks:</b> Sustained CPS (Calls Per Second), bursts, and ramp-up testing to detect server breaking points.", bullet_style))
story.append(Spacer(1, 6))

# Section 5: Executive Justification
story.append(KeepTogether([
    Paragraph("5. Executive Justification for Engineering Leadership", h1_style),
    Paragraph("<b>1. Eliminates Production Surprises:</b> Voice AI operates under strict real-time WebRTC constraints. Packet loss, DTLS handshakes, and LLM rate-limits only emerge under real traffic. This framework catches degradation before real users experience it.", bullet_style),
    Paragraph("<b>2. Isolates Component Bottlenecks:</b> Segmented metrics reveal exactly which layer is slow: the API gateway, WebRTC media server, Speech-to-Text (STT), LLM reasoning, or Text-to-Speech (TTS).", bullet_style),
    Paragraph("<b>3. 100% Auditable Proof:</b> Every test produces an indisputable synchronized stereo WAV recording (Left = Caller, Right = Agent) and JSON telemetry, giving leadership verifiable evidence of SLA compliance.", bullet_style),
    Paragraph("<b>4. CI/CD Integration Ready:</b> Automated benchmarking can run in deployment pipelines to gate releases that fail conversational latency targets (&lt;1.5s).", bullet_style),
]))

doc.build(story)
print(f"PDF generated successfully: {Path(pdf_path).resolve()}")
