import os, json, base64
from typing import Any, Dict, List

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

# PDF + Google Drive
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.pdfgen import canvas
from googleapiclient.discovery import build
from google.oauth2.service_account import Credentials
from googleapiclient.http import MediaFileUpload

# =======================
# Setup
# =======================
load_dotenv()  # reads .env in this folder
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL_VISION_TEXT = "gpt-4o-mini"  # Vision + Text model

app = FastAPI(title="Vikas ENT AI – OCR + Analysis")

# Allow local devices (your phone) to call the server; tighten origins later for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =======================
# Complaint Profiles + Helpers
# =======================
COMPLAINT_PROFILES = {
    "Vertigo": {
        "required_keys": [
            "SPINNING SENSATION", "IMBALANCE WITHOUT SPINNING", "TRIGGERS",
            "HEARING LOSS", "TINNITUS (RINGING)", "AURAL FULLNESS / PRESSURE",
            "RECENT VIRAL ILLNESS (PAST 4 WEEKS)", "HEADACHE WITH VERTIGO",
            "NEUROLOGICAL SYMPTOMS (DOUBLE VISION / WEAKNESS)"
        ],
        "red_flags": [
            "NEW NEUROLOGICAL DEFICITS",
            "SEVERE HEADACHE / NECK PAIN",
            "SEVERE CONTINUOUS VERTIGO > 24 HOURS",
            "RECENT HEAD TRAUMA",
        ],
        "rules": {
            "BPPV": [
                ("SPINNING SENSATION", "YES", 0.25),
                ("TRIGGERS", "TURNING IN BED", 0.35),
                ("DIX-HALLPIKE", "POSITIVE", 0.40),
            ],
            "Ménière’s disease": [
                ("HEARING LOSS", "YES", 0.35),
                ("TINNITUS (RINGING)", "YES", 0.25),
                ("AURAL FULLNESS / PRESSURE", "YES", 0.25),
            ],
            "Vestibular neuritis": [
                ("RECENT VIRAL ILLNESS (PAST 4 WEEKS)", "YES", 0.35),
                ("SPINNING SENSATION", "YES", 0.20),
                ("IMBALANCE WITHOUT SPINNING", "YES", 0.15),
            ],
            "Vestibular migraine": [
                ("HEADACHE WITH VERTIGO", "YES", 0.30),
                ("PHOTOPHOBIA / PHONOPHOBIA", "YES", 0.30),
                ("MIGRAINE HISTORY (PERSONAL/FAMILY)", "YES", 0.30),
            ],
        },
        "default_tests": [
            "Dix–Hallpike", "Supine roll test", "HINTS bedside exam",
            "Pure tone audiogram (if auditory symptoms)"
        ],
    },
    "Nasal Obstruction": {
        "required_keys": [
            "Side predominance", "Duration", "Allergy history", "Watery rhinorrhea",
            "Thick discharge", "Facial pain/pressure", "Reduced smell",
            "Epistaxis (nosebleed)", "Previous nasal trauma/surgery"
        ],
        "red_flags": [
            "unilateral persistent block", "unilateral blood-stained discharge",
            "facial numbness", "eye swelling", "weight loss"
        ],
        "rules": {
            "Allergic rhinitis": [("Allergy history", "YES", 0.5), ("Watery rhinorrhea", "YES", 0.3)],
            "Chronic rhinosinusitis +/- polyps": [("Thick discharge", "YES", 0.3), ("Facial pain/pressure", "YES", 0.3), ("Reduced smell", "YES", 0.3)],
            "DNS/turbinate hypertrophy": [("Side predominance", "UNILATERAL", 0.4), ("Previous nasal trauma/surgery", "YES", 0.3)],
        },
        "default_tests": ["Anterior rhinoscopy / endoscopy", "CT PNS if indicated", "Allergy testing if relevant"],
    },
    "Lump in Throat": {
        "required_keys": [
            "Duration", "Dysphagia", "Odynophagia", "Weight loss",
            "Hoarseness", "Reflux symptoms", "Neck mass"
        ],
        "red_flags": ["progressive dysphagia", "odynophagia", "weight loss", "neck mass", "voice change"],
        "rules": {
            "Globus pharyngeus": [("Reflux symptoms", "YES", 0.4), ("Intermittent symptoms", "YES", 0.2)],
            "Laryngopharyngeal reflux": [("Reflux symptoms", "YES", 0.5), ("Hoarseness", "YES", 0.2)],
        },
        "default_tests": ["Flexible nasoendoscopy", "Reflux workup if suggested"],
    },
}

def normalize_key(s: str) -> str:
    return " ".join(s.upper().split())

def score_rules(parsed: dict, profile: dict) -> List[Dict[str, any]]:
    scores = {}
    for dx, conds in profile.get("rules", {}).items():
        total = 0.0
        for key, wanted, w in conds:
            v = str(parsed.get(normalize_key(key), "")).upper()
            wanted = wanted.upper()
            if wanted in v:
                total += w
        scores[dx] = min(total, 0.99)
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [{"dx": k, "score": round(v, 2)} for k, v in ranked if v > 0]

def enrich_management(dx_list: List[Dict[str, any]], profile: dict) -> List[Dict[str, any]]:
    out = []
    for item in dx_list:
        dx = item["dx"]
        if dx == "BPPV":
            item.update({
                "signs": ["Positional nystagmus"],
                "tests": ["Dix–Hallpike", "Supine roll test"],
                "notes": "Consider canalith repositioning (Epley)."
            })
        elif dx == "Ménière’s disease":
            item.update({
                "signs": ["Fluctuating SNHL", "Aural fullness", "Tinnitus"],
                "tests": ["Pure tone audiogram", "ECochG (if available)"],
                "notes": "Salt restriction; vestibular suppressants short-term."
            })
        elif dx == "Vestibular neuritis":
            item.update({
                "signs": ["Unidirectional horizontal nystagmus"],
                "tests": ["HINTS", "Head impulse test"],
                "notes": "Early vestibular rehab; exclude stroke if atypical."
            })
        elif dx == "Vestibular migraine":
            item.update({
                "signs": ["Photophobia/phonophobia during attacks"],
                "tests": ["Migraine diary", "Audiogram if auditory sx"],
                "notes": "Migraine lifestyle + prophylaxis if frequent."
            })
        else:
            item.setdefault("signs", [])
            item.setdefault("tests", [])
            item.setdefault("notes", "")
        out.append(item)
    if not out:
        out = [{
            "dx": "Non-specific",
            "score": 0.3,
            "signs": [],
            "tests": profile.get("default_tests", []),
            "notes": "Gather more history / exam."
        }]
    return out

def summarize_next_steps(profile: dict, parsed: dict, diffs: List[Dict[str, any]]) -> List[str]:
    steps = []
    missing = []
    for k in profile.get("required_keys", []):
        if normalize_key(k) not in parsed:
            missing.append(k)
    if missing:
        steps.append(f"Collect missing key answers: {', '.join(missing[:6])}" + ("..." if len(missing)>6 else ""))
    if profile.get("default_tests"):
        steps.append("Initial tests: " + ", ".join(profile["default_tests"]))
    if diffs:
        top = diffs[0]
        if top.get("tests"):
            steps.append(f"Priority to confirm '{top['dx']}': " + ", ".join(top["tests"]))
    return steps

def extract_red_flags(profile: dict, parsed: dict) -> List[str]:
    rf = []
    text = " ".join([f"{k}: {v}" for k, v in parsed.items()]).upper()
    for flag in profile.get("red_flags", []):
        if flag.upper() in text:
            rf.append(flag)
    return rf

# =======================
# Schemas for /analyze
# =======================
class Patient(BaseModel):
    name: str = ""
    age: str = ""
    complaint: str = ""

class Payload(BaseModel):
    patient: Patient
    responses: Dict[str, Any] = {}

# =======================
# OpenAI helpers
# =======================
def openai_vision_ocr(image_bytes: bytes) -> str:
    """Extract raw text from image via OpenAI vision model."""
    if not client.api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not configured on server.")
    try:
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        resp = client.chat.completions.create(
            model=MODEL_VISION_TEXT,
            messages=[
                {
                    "role": "system",
                    "content": "You are an OCR engine. Extract all readable text as plain UTF-8. Preserve line breaks. Output only text."
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Extract raw text from this photo. Keep formatting and line breaks."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
                    ]
                }
            ],
            temperature=0
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        msg = str(e)
        if "insufficient_quota" in msg or "rate limit" in msg.lower():
            raise HTTPException(status_code=429, detail="Quota/rate limit from OpenAI. Check billing/limits.")
        raise HTTPException(status_code=502, detail=f"OCR error: {msg}")

def openai_clinical_analyze(raw_text: str, complaint_hint: str = "Vertigo") -> Dict[str, Any]:
    """AI parsing + local scoring/enrichment + next steps/red flags."""
    profile = COMPLAINT_PROFILES.get(complaint_hint, COMPLAINT_PROFILES["Vertigo"])

    prompt = f"""
You are an ENT decision-support assistant.

OCR raw text (complaint: {complaint_hint}):
---
{raw_text}
---

TASKS:
1) Parse into key:value fields (keys in UPPERCASE, concise; split on ':' or '-' if field-like).
2) Draft a provisional diagnosis string.
3) Provide up to 5 differentials.

Return STRICT JSON only:

{{
  "parsed_fields": {{
    "KEY": "value"
  }},
  "provisional": "",
  "differentials": [
    {{"dx":"", "score": 0.0, "signs":[], "tests":[], "notes":""}}
  ]
}}
""".strip()

    try:
        resp = client.chat.completions.create(
            model=MODEL_VISION_TEXT,
            messages=[
                {"role": "system", "content": "Return only valid JSON. No commentary."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        content = resp.choices[0].message.content.strip().strip("`")
        content = content.replace("json\n", "").replace("JSON\n", "")
    except Exception as e:
        msg = str(e)
        if "insufficient_quota" in msg or "rate limit" in msg.lower():
            raise HTTPException(status_code=429, detail="Quota/rate limit from OpenAI. Check billing/limits.")
        raise HTTPException(status_code=502, detail=f"OpenAI error: {msg}")

    # Try parse; if fail, start with empty shell
    try:
        data = json.loads(content)
    except Exception:
        data = {"parsed_fields": {}, "provisional": "", "differentials": []}

    # Normalize keys to UPPERCASE single-spaced
    parsed = {}
    for k, v in (data.get("parsed_fields") or {}).items():
        parsed[normalize_key(k)] = str(v)

    # Rule-based scoring and enrichment for stability/consistency
    ranked = score_rules(parsed, profile)
    enriched = enrich_management(ranked, profile)

    # Red flags + next steps
    red_flags = extract_red_flags(profile, parsed)
    next_steps = summarize_next_steps(profile, parsed, enriched)

    return {
        "parsed_fields": parsed,
        "provisional": data.get("provisional", ""),
        "differentials": enriched,
        "red_flags": red_flags,
        "next_steps": next_steps,
    }

# =======================
# PDF + Google Drive helpers
# =======================
def _bullet_wrap(c: canvas.Canvas, x, y, text, max_width):
    # simple wrap for bullets
    from reportlab.pdfbase.pdfmetrics import stringWidth
    words = (text or "").split()
    line = ""
    while words:
        nxt = (line + " " + words[0]).strip()
        if stringWidth(nxt, "Helvetica", 11) <= max_width:
            line = nxt
            words.pop(0)
        else:
            c.drawString(x, y, "• " + line)
            y -= 14
            line = ""
    if line:
        c.drawString(x, y, "• " + line); y -= 14
    return y

def generate_pdf(patient: dict, ai: dict, raw_text: str, img_bytes: bytes | None) -> str:
    # returns absolute file path
    from datetime import datetime
    os.makedirs("out", exist_ok=True)
    filename = f"out/{(patient.get('name') or 'patient').replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    c = canvas.Canvas(filename, pagesize=A4)
    W, H = A4
    x, y = 2*cm, H - 2*cm

    # Header
    c.setFont("Helvetica-Bold", 16); c.drawString(x, y, "Vikas ENT — AI Analysis"); y -= 22
    c.setFont("Helvetica", 11)
    c.drawString(x, y, f"Patient: {patient.get('name','')}   Age: {patient.get('age','')}   Sex: {patient.get('sex','')}   Mobile: {patient.get('mobile','')}")
    y -= 16
    c.drawString(x, y, f"Chief Complaint: {patient.get('complaint','')}"); y -= 22

    # Optional thumbnail
    if img_bytes:
        import tempfile
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        tmp.write(img_bytes); tmp.close()
        try:
            c.drawImage(tmp.name, x, y-120, width=5*cm, height=7*cm, preserveAspectRatio=True, mask='auto')
            y -= 130
        except:
            pass

    def title(txt):
        nonlocal y
        if y < 4*cm: c.showPage(); y = H - 2*cm
        c.setFont("Helvetica-Bold", 13); c.drawString(x, y, txt); y -= 18
        c.setFont("Helvetica", 11)

    title("Provisional Diagnosis")
    c.drawString(x, y, (ai.get("provisional") or "—")); y -= 18

    title("Differential Diagnosis")
    diffs = ai.get("differentials") or []
    if not diffs: c.drawString(x, y, "—"); y -= 14
    for d in diffs:
        line = f"{d.get('dx','')}  (score: {d.get('score','')})"
        y = _bullet_wrap(c, x, y, line, W-3*cm)

    title("Clinical / Lab tests to be done")
    tests = []
    for d in diffs: tests += d.get("tests", [])
    tests = list(dict.fromkeys(tests))  # unique
    if not tests: c.drawString(x, y, "—"); y -= 14
    for t in tests: y = _bullet_wrap(c, x, y, t, W-3*cm)

    title("Keep in Mind")
    keep = (ai.get("red_flags") or []) + (ai.get("next_steps") or [])
    if not keep: c.drawString(x, y, "—"); y -= 14
    for k in keep: y = _bullet_wrap(c, x, y, k, W-3*cm)

    title("Treatment advised")
    tx = ""
    if diffs and (diffs[0].get("notes")): tx = diffs[0]["notes"]
    c.drawString(x, y, tx or "—"); y -= 18

    title("Raw OCR Text (for record)")
    for para in (raw_text or "—").splitlines():
        y = _bullet_wrap(c, x, y, para, W-3*cm)

    c.showPage(); c.save()
    return os.path.abspath(filename)

def drive_service():
    key = os.getenv("GOOGLE_DRIVE_CREDENTIALS")
    if not key or not os.path.exists(key):
        raise HTTPException(status_code=500, detail="Google Drive credentials not configured.")
    scopes = ["https://www.googleapis.com/auth/drive.file"]
    creds = Credentials.from_service_account_file(key, scopes=scopes)
    return build("drive", "v3", credentials=creds)

def upload_to_drive(file_path: str, folder_id: str | None) -> dict:
    svc = drive_service()
    meta = {"name": os.path.basename(file_path)}
    if folder_id:
        meta["parents"] = [folder_id]
    media = MediaFileUpload(file_path, mimetype="application/pdf", resumable=False)
    res = svc.files().create(
        body=meta, media_body=media, fields="id, webViewLink, webContentLink"
    ).execute()
    return {
        "id": res.get("id"),
        "webViewLink": res.get("webViewLink"),
        "webContentLink": res.get("webContentLink"),
    }

# =======================
# Endpoints
# =======================
@app.get("/health")
def health():
    if not client.api_key:
        return {"ok": False, "error": "OPENAI_API_KEY not set on server"}
    return {"ok": True}

@app.post("/ocr_analyze")
async def ocr_analyze(file: UploadFile = File(...), complaint: str = "Vertigo"):
    """
    Upload a questionnaire photo once; server does OCR + clinical analysis.
    Returns: { raw_text, parsed_fields, provisional, differentials[], red_flags[], next_steps[] }
    """
    try:
        image_bytes = await file.read()
        raw = openai_vision_ocr(image_bytes)
        result = openai_clinical_analyze(raw_text=raw, complaint_hint=complaint)
        result["raw_text"] = raw
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {e}")

class AnalyzePayload(BaseModel):
    patient: Patient
    responses: Dict[str, Any] = {}

@app.post("/analyze")
def analyze(p: AnalyzePayload):
    """
    Optional endpoint if you already have structured fields on the app side.
    Produces differentials + tests from key:value pairs.
    """
    txt = " ".join([f"{k}: {v}" for k, v in (p.responses or {}).items()])
    raw = f"Complaint: {p.patient.complaint}\nAge: {p.patient.age}\n{txt}"
    return openai_clinical_analyze(raw_text=raw, complaint_hint=p.patient.complaint or "ENT")

@app.post("/export_pdf")
def export_pdf(
    patient: dict = Body(...),
    ai: dict = Body(...),
    raw_text: str = Body(""),
    image_b64: str | None = Body(None)
):
    """
    Build a PDF from {patient, ai, raw_text} (+ optional image) and upload to Google Drive.
    Returns {ok, file:{id, webViewLink, webContentLink}, pdf_path}.
    """
    try:
        img_bytes = base64.b64decode(image_b64) if image_b64 else None
        pdf_path = generate_pdf(patient, ai, raw_text, img_bytes)
        folder = os.getenv("DRIVE_FOLDER_ID")
        meta = upload_to_drive(pdf_path, folder)
        return {"ok": True, "file": meta, "pdf_path": pdf_path}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export error: {e}")
