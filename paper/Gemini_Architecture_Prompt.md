# Gemini Prompt — mlcompass System Architecture Diagram

**How to use this:**
1. Open <https://gemini.google.com> (or Imagen)
2. Paste the entire **PROMPT TO PASTE** block below into Gemini
3. Download the generated PNG/JPG
4. Save it as `architecture_diagram.png` in this folder
5. Open `Capstone_Report.docx` in Word
6. Find the page that says *"[ INSERT SYSTEM ARCHITECTURE DIAGRAM HERE ]"*
7. Delete that placeholder, then **Insert → Picture** → pick `architecture_diagram.png`
8. Right-click the image → **Wrap Text → In Line with Text** → centre it

If Gemini's first attempt isn't clean, ask for a regeneration with the
follow-up:

> *Regenerate the same diagram with cleaner spacing, larger labels, and
> arrows that don't overlap any box.*

---

## PROMPT TO PASTE

```
Create a modern, clean, professional infographic diagram showing the system
architecture of "mlcompass" — an AI assistant for machine learning
practitioners. The diagram should be a top-down flow chart with three
distinct horizontal layers.

LAYER 1 (TOP) — "USER-FACING SURFACES" — three boxes side by side:
- Left box: "CLI" with a terminal/command-prompt icon. Caption:
  "Terminal commands: mlcompass advise / audit / evaluate / ..."
- Middle box: "MCP Server" with a chat-bubble icon. Caption:
  "Used inside Claude Desktop, Cursor, Continue, Claude Code —
  the user just talks in natural language"
- Right box: "Self-Driving Agent" with a small robot icon. Caption:
  "mlcompass agent — drives the pipeline end-to-end on its own"

LAYER 2 (MIDDLE) — "THE BRAIN" — two stacked boxes:
- Top box (soft blue background): "Deterministic Tool Layer". Caption:
  "Pure Python. No AI involved. Reads the dataset, parses the script,
  scans the log, computes the metrics. Always the same answer for the
  same input."
- Bottom box (soft purple background): "Optional LLM Narrator". Caption:
  "Claude AI explains the structured output in plain English under a
  strict 'anti-hallucination' contract — it can only cite facts the
  tool layer found."

LAYER 3 (BOTTOM) — "PERSISTENT MEMORY" — one wide box:
- Title: ".mlcompass/ project context folder"
- Sub-labels: "project.yaml • context.json • datasets/ • runs/ •
  advice.log"
- Caption: "Remembers every decision the assistant has made across days"

ARROWS:
- All three top boxes (CLI, MCP, Agent) flow down into the Deterministic
  Tool Layer.
- The Tool Layer flows down into the LLM Narrator (label this arrow:
  "structured evidence dict").
- The LLM Narrator flows back upward to all three surfaces (label this
  arrow: "plain-English answer").
- The Persistent Memory box connects to the Tool Layer with a
  bidirectional arrow (label: "read / write").

STYLE GUIDELINES:
- Color palette: Soft blue (#3B82F6), soft purple (#8B5CF6),
  neutral gray (#6B7280) for arrows, white background.
- Typography: Modern sans-serif (Inter, Roboto, or similar), highly
  readable, no thin fonts.
- Icons: Minimalist line icons (Heroicons-style).
- Layout: Centred, generous whitespace, rounded corners on every box,
  soft drop shadows.
- Title at the top: "mlcompass — System Architecture".
- Aspect ratio: 4:3 (will be inserted into a portrait A4 Word document).

GOAL — a beginner with no ML background should be able to look at this
diagram for 30 seconds and immediately understand:
1. The same tool can be used three different ways (terminal, in-chat,
   autonomous).
2. The "brain" has two clearly-separated layers — facts come from pure
   code, narration comes from AI on top.
3. The AI never invents facts; it can only describe what the code found.
4. The system remembers what it decided yesterday so it doesn't start
   from scratch every day.

CONSTRAINT — do NOT include any code, JSON, command-line snippets, or
file paths inside the diagram itself. Pure conceptual flow only.
```

---

## Alternative if Gemini struggles with the full prompt

Use this shorter version for a second attempt:

```
A modern infographic titled "mlcompass — System Architecture".
Three horizontal layers, top to bottom:

1. Three small boxes side by side: "CLI (Terminal)", "MCP Server
   (Chat)", "Self-Driving Agent".
2. Two stacked boxes labelled "Deterministic Tool Layer (pure
   Python)" and below it "Optional LLM Narrator (Claude AI)".
3. One wide box labelled ".mlcompass/ Persistent Memory".

Arrows connect every top box to the Tool Layer; the Tool Layer
sends data to the LLM Narrator; the LLM Narrator answers all three
surfaces; the Persistent Memory connects to the Tool Layer with a
two-way arrow.

Soft blue + soft purple + gray colour palette, rounded boxes,
clean sans-serif font, minimalist icons, white background, 4:3.
Professional, beginner-friendly. No code snippets inside the
diagram.
```
