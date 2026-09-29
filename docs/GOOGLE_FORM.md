# Google Form — Sri Lankan SMS collection

Create at **forms.google.com → Blank form**. Copy each block below.
**Keep the question titles exactly as written** — `src/import_form.py` reads them.

Settings (⚙ Settings tab):
- Responses → **Collect email addresses: OFF** (keeps it anonymous)
- Responses → **Limit to 1 response: OFF** (people can submit many messages)
- Presentation → Confirmation message: *"Thank you! Submit another message if you have more."*
- Presentation → **Show link to submit another response: ON**

---

## Form title
**Help stop SMS scams in Sri Lanka 🇱🇰 | SMS වංචා නවත්වන්න උදව් කරන්න | SMS மோசடிகளை நிறுத்த உதவுங்கள்**

## Form description
```
I am a final-year student at SLIIT building an AI system that detects scam (phishing) SMS in English, Sinhala and Tamil. Please share SMS messages you have received — both scam messages AND normal messages (OTP, bank alerts, delivery, bills, personal).

• One message per submission — you can submit as many as you like.
• BEFORE pasting, remove private details: replace names with <NAME>, phone numbers with <PHONE>, account/card numbers with <ACC>.
• Please do NOT click any link inside a scam message.
• No email or name is collected. Messages are used only for academic research.

SLIIT අවසන් වසරේ ශිෂ්‍යයෙකු ලෙස, ඉංග්‍රීසි, සිංහල සහ දෙමළ වංචනික SMS හඳුනාගන්නා AI පද්ධතියක් හදනවා. ඔබට ලැබුණු වංචනික සහ සාමාන්‍ය SMS බෙදාගන්න. නම්, දුරකථන අංක, ගිණුම් අංක ඉවත් කරන්න. ලින්ක් click කරන්න එපා.

SLIIT இறுதியாண்டு மாணவனாக, ஆங்கிலம், சிங்களம், தமிழ் மோசடி SMS-களைக் கண்டறியும் AI அமைப்பை உருவாக்குகிறேன். உங்களுக்கு வந்த மோசடி மற்றும் சாதாரண SMS-களைப் பகிருங்கள். பெயர், தொலைபேசி, கணக்கு எண்களை நீக்குங்கள். இணைப்புகளைக் கிளிக் செய்ய வேண்டாம்.
```

---

## Questions

### Q1 — Checkboxes · Required
**Title:** `Consent`
- ☐ I agree that this message can be used anonymously for academic research, and I have removed personal details.

### Q2 — Paragraph · Required
**Title:** `Message text`
**Description:** Copy-paste the full SMS exactly as received (keep links and spelling). Replace names/numbers with <NAME>, <PHONE>, <ACC>.

### Q3 — Multiple choice · Required
**Title:** `Message type`
- Scam / phishing — asks to click a link, pay a fee, share OTP/PIN, or claims a prize/fine/blocked account
- Promotion / advertisement — genuine offer from a real company
- Normal / legitimate — OTP, bank alert, bill, delivery, personal message
- Not sure

### Q4 — Multiple choice · Required
**Title:** `Language`
- English
- Sinhala (සිංහල අකුරු)
- Tamil (தமிழ் எழுத்து)
- Singlish (Sinhala in English letters, e.g. "oyage account eka")
- Tanglish (Tamil in English letters, e.g. "unga account")
- Mixed script (e.g. Sinhala/Tamil letters + English words)

### Q5 — Multiple choice · Optional
**Title:** `Sender`
- Bank
- Mobile operator (Dialog / Mobitel / Hutch / Airtel)
- Courier / Post
- Government
- Shop / company
- Unknown number
- Known person (friend / family)
- Other

### Q6 — Short answer · Optional
**Title:** `Month received`
**Description:** e.g. 2026-08 (approximate is fine)

### Q7 — Short answer · Optional
**Title:** `How did you know it was a scam?`
**Description:** e.g. bank confirmed, fake link, asked for OTP (skip for normal messages)

---

## Sharing plan (aim: ≥ 200 responses in 2 weeks)
1. SLIIT batch / faculty WhatsApp groups + class Teams channels
2. Family WhatsApp groups (parents receive the most Sinhala/Tamil scams)
3. Ask 3–4 friends to fill 10+ messages each straight from their SMS inbox
4. Share again after 1 week with a count: "We have 120 so far — help us reach 200!"

## After collecting
Responses tab → ⋮ → **Download responses (.csv)** → save as
`data/real_sms/form_responses.csv`, then run:

```
python -m src.import_form            # -> data/real_sms/real_test.csv (anonymised, deduplicated)
python -m src.data_prep --check-real  # validate
```
