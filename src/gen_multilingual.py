"""Step 4 - Generate the Sri Lankan multilingual SMS training set.

Hand-written message patterns (templates) in Sinhala, Tamil, Singlish and
Tanglish, filled with varied banks, couriers, amounts, links, deadlines ...
Every row keeps its `template_id` so data_prep.py can split by TEMPLATE
(all variants of one pattern go to the same side) - no near-copy leakage.

    python -m src.gen_multilingual            # -> data/raw/multilingual_train.csv

columns: text,label,language,script,category,origin,template_id
"""
from __future__ import annotations

import csv
import random
import string
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "multilingual_train.csv"
SEED = 42
VARIANTS = 8          # fills per template
rng = random.Random(SEED)

# ------------------------------------------------------------------ slot values
BANK = ["BOC", "Peoples Bank", "Commercial Bank", "Sampath Bank", "HNB", "NSB", "NDB", "Seylan Bank", "DFCC", "Pan Asia Bank", "Union Bank", "Cargills Bank"]
BANKDOM = ["boc", "peoplesbank", "combank", "sampath", "hnb", "nsb", "ndb", "seylan", "dfcc"]
COURIER = ["SL Post", "DHL", "FedEx", "Aramex", "Pronto", "Domex", "Koombiyo"]
TELCO = ["Dialog", "Mobitel", "Hutch", "Airtel"]
SHOP = ["Keells", "Cargills", "Daraz", "Glomark", "Arpico", "ODEL", "Singer", "Abans", "Softlogic", "Kapruka"]
WALLET = ["eZ Cash", "mCash", "Genie", "FriMi", "iPay"]
GOV = ["GovPay", "Aswesuma", "CEB", "Water Board", "Inland Revenue", "RMV", "Sri Lanka Customs"]
TLD = ["top", "xyz", "buzz", "icu", "site", "online", "help", "cc", "shop", "info", "live", "club", "click", "link", "vip"]
WORD = ["verify", "secure", "update", "kyc", "help", "pay", "refund", "gift", "reward", "login", "service", "support", "confirm", "track"]
SHORT = ["bit.ly", "tinyurl.com", "cutt.ly", "is.gd", "rb.gy", "t.ly", "shorturl.at"]
CITY = ["Colombo", "Kandy", "Galle", "Jaffna", "Batticaloa", "Kurunegala", "Matara", "Negombo", "Trincomalee", "Anuradhapura", "Badulla", "Ratnapura"]
CITY_SI = ["කොළඹ", "මහනුවර", "ගාල්ල", "යාපනය", "කුරුණෑගල", "මාතර", "මීගමුව", "අනුරාධපුර", "බදුල්ල", "රත්නපුර"]
CITY_TA = ["கொழும்பு", "கண்டி", "காலி", "யாழ்ப்பாணம்", "மட்டக்களப்பு", "திருகோணமலை", "குருநாகல்", "நீர்கொழும்பு", "வவுனியா", "மன்னார்"]
HOSP = ["Asiri", "Nawaloka", "Lanka Hospitals", "Durdans", "Hemas", "Ninewells"]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def rs(lo=500, hi=250000, step=50):
    return f"{rng.randrange(lo, hi, step):,}"

def cents():
    return f"{rng.randrange(100, 150000):,}.{rng.randrange(0, 100):02d}"

def otp():
    return str(rng.randrange(100000, 999999))

def code(n=6):
    return "".join(rng.choices(string.ascii_letters + string.digits, k=n))

def link(brand: str | None = None) -> str:
    r = rng.random()
    if r < 0.35:
        return f"{rng.choice(SHORT)}/{code(rng.randint(5, 7))}"
    b = brand or rng.choice(BANKDOM + ["slpost", "govpay", "dialog", "mobitel", "daraz", "customs", "ceb", "aswesuma"])
    host = rng.choice([f"{b}-{rng.choice(WORD)}", f"{b}{rng.choice(WORD)}", f"{rng.choice(WORD)}-{b}-lk", f"{b}-lk"])
    url = f"{host}.{rng.choice(TLD)}"
    if rng.random() < 0.4:
        url += "/" + rng.choice(["lk", "pay", "verify", "login", code(4)])
    return ("https://" if rng.random() < 0.3 else "http://" if rng.random() < 0.2 else "") + url

def phone():
    return "07" + rng.choice("0125678") + str(rng.randrange(1000000, 9999999))

def hrs():
    return rng.choice(["12", "24", "48", "2", "6"])

def date():
    return f"{rng.randint(1, 28)} {rng.choice(MONTHS)}"

def time_():
    return f"{rng.randint(1, 12)}.{rng.choice(['00', '15', '30', '45'])}"

def acct():
    return "xxxx" + str(rng.randrange(1000, 9999))

SLOTS = {
    "BANK": lambda: rng.choice(BANK), "COURIER": lambda: rng.choice(COURIER), "TELCO": lambda: rng.choice(TELCO),
    "SHOP": lambda: rng.choice(SHOP), "WALLET": lambda: rng.choice(WALLET), "GOV": lambda: rng.choice(GOV),
    "AMT": rs, "SAMT": lambda: rs(100, 3000, 50), "BAMT": lambda: rs(100000, 5000000, 10000), "CENTS": cents,
    "OTP": otp, "LINK": link, "PHONE": phone, "HRS": hrs, "DATE": date, "TIME": time_, "ACC": acct,
    "GB": lambda: rng.choice(["5", "10", "25", "50", "100"]), "PCT": lambda: rng.choice(["10", "15", "20", "25", "30", "40", "50"]),
    "CITY": lambda: rng.choice(CITY), "CITY_SI": lambda: rng.choice(CITY_SI), "CITY_TA": lambda: rng.choice(CITY_TA),
    "HOSP": lambda: rng.choice(HOSP), "MIN": lambda: str(rng.choice([5, 10, 15, 20, 30])),
    "WA": lambda: "947" + rng.choice("0125678") + str(rng.randrange(1000000, 9999999)),
    "TRK": lambda: "LK" + str(rng.randrange(10000000, 99999999)),
}

def fill(t: str) -> str:
    out = t
    for k, f in SLOTS.items():
        while "{" + k + "}" in out:
            out = out.replace("{" + k + "}", f(), 1)
    return out

# ------------------------------------------------------------------ templates
# (language, script, label, category, [templates])
T: list[tuple[str, str, str, str, list[str]]] = []

# ============================ SINHALA ============================
T += [("si", "native", "smishing", "bank", [
 "{BANK}: ඔබගේ ගිණුම තාවකාලිකව අත්හිටුවා ඇත. පැය {HRS}ක් ඇතුළත තහවුරු කරන්න: {LINK}",
 "ප්‍රිය පාරිභෝගිකය, ඔබගේ {BANK} කාඩ්පත අවහිර වී ඇත. නැවත සක්‍රීය කිරීමට OTP අංකය මෙම ලින්ක් එකේ ඇතුළත් කරන්න {LINK}",
 "{BANK}: ඔබගේ KYC විස්තර යාවත්කාලීන කර නැත. අද දිනය තුළ යාවත්කාලීන නොකළහොත් ගිණුම වසා දමනු ලැබේ. {LINK}",
 "ඔබගේ ගිණුමෙන් රු. {AMT}ක් ආපසු ගැනීමට ඉල්ලීමක් ලැබී ඇත. එය ඔබ නොවේ නම් වහාම අවලංගු කරන්න: {LINK}",
 "ඔබගේ ක්‍රෙඩිට් කාඩ් ප්‍රසාද ලකුණු {AMT}ක් අද කල් ඉකුත් වේ. මුදල් බවට පරිවර්තනය කරන්න: {LINK}",
 "{BANK}: සැක සහිත පිවිසීමක් හඳුනාගෙන ඇත. ගිණුම ආරක්ෂා කිරීමට ඔබට ලැබුණු කේතය අපට reply කරන්න.",
 "ඔබගේ ඩෙබිට් කාඩ්පත කල් ඉකුත් වී ඇත. නව කාඩ්පත සඳහා විස්තර ඇතුළත් කරන්න: {LINK}",
 "ඔබගේ ණය ඉල්ලුම්පත අනුමත විය! රු. {BAMT} ලබා ගැනීමට සැකසුම් ගාස්තුව රු. {SAMT} මෙම ගිණුමට තැන්පත් කරන්න.",
 "අවවාදයයි! ඔබගේ {BANK} ඔන්ලයින් බැංකු ගිණුමට වෙනත් උපාංගයකින් පිවිසීමට උත්සාහ කර ඇත. වහාම මුරපදය වෙනස් කරන්න: {LINK}",
 "ඔබගේ ගිණුමට රු. {AMT}ක් වැරදීමකින් බැර වී ඇත. එය ආපසු ලබා දීමට මෙම ලින්ක් එකෙන් තහවුරු කරන්න {LINK}",
 ]), ("si", "native", "smishing", "delivery", [
 "{COURIER}: ඔබගේ පාර්සලය ලිපිනය අසම්පූර්ණ නිසා රඳවා තබා ඇත. රු. {SAMT} ගාස්තුව ගෙවා යාවත්කාලීන කරන්න: {LINK}",
 "ඔබගේ පැකේජය රේගුවේ රඳවා ඇත. නිදහස් කිරීමට රේගු ගාස්තුව රු. {SAMT} අද ගෙවන්න: {LINK}",
 "{COURIER}: බෙදාහැරීම අසාර්ථක විය. පැය {HRS}ක් ඇතුළත නැවත දිනයක් තෝරන්න නැතහොත් පාර්සලය ආපසු යවනු ලැබේ {LINK}",
 "ඔබගේ {SHOP} ඇණවුම නැවතී ඇත. ගෙවීම තහවුරු කිරීමට කාඩ් විස්තර ඇතුළත් කරන්න: {LINK}",
 ]), ("si", "native", "smishing", "government", [
 "රථවාහන දඩ මුදල රු. {SAMT} ගෙවා නැත. අද ගෙවා නොමැති නම් රියදුරු බලපත්‍රය අත්හිටුවනු ලැබේ: {LINK}",
 "අස්වැසුම ප්‍රතිලාභ මුදල රු. {AMT} ඔබගේ නමට අනුමත වී ඇත. ලබා ගැනීමට බැංකු විස්තර ඇතුළත් කරන්න: {LINK}",
 "විදුලිබල මණ්ඩලය: ඔබගේ බිල්පත ගෙවා නැති නිසා අද රාත්‍රී විදුලිය විසන්ධි කෙරේ. වහාම ගෙවන්න {LINK}",
 "ඔබගේ ජාතික හැඳුනුම්පත් අංකය යාවත්කාලීන කළ යුතුය, නැතහොත් SIM පත අක්‍රිය වේ. මෙතනින් යාවත්කාලීන කරන්න: {LINK}",
 "ආදායම් බදු ආපසු ගෙවීම රු. {AMT} ඔබට හිමියි. පැය {HRS}ක් ඇතුළත ඉල්ලුම් කරන්න: {LINK}",
 ]), ("si", "native", "smishing", "prize", [
 "සුබ පැතුම්! {TELCO} ජයග්‍රාහකයා ලෙස ඔබ රු. {BAMT}ක් දිනා ඇත. ත්‍යාගය ලබා ගැනීමට {PHONE} අමතන්න.",
 "ඔබගේ අංකය සංවත්සර ලොතරැයියෙන් තෝරාගෙන ඇත! ත්‍යාගය ලබා ගැනීමට ලියාපදිංචි ගාස්තුව රු. {SAMT} {WALLET} කරන්න.",
 "{TELCO}: ඔබට නොමිලේ {GB}GB දත්ත ලැබී ඇත. අද දිනය තුළ සක්‍රීය කරන්න: {LINK}",
 "{SHOP} ත්‍යාග කූපනය රු. {AMT}ක් ඔබට! සමීක්ෂණය පුරවා ලබා ගන්න: {LINK}",
 ]), ("si", "native", "smishing", "job", [
 "නිවසේ සිට වැඩ කර දිනකට රු. {AMT} උපයන්න. WhatsApp: wa.me/{WA}",
 "අර්ධකාලීන රැකියාවක්: වීඩියෝ like කිරීමට දිනකට රු. {AMT}. දැන්ම සම්බන්ධ වන්න {LINK}",
 "විදේශ රැකියා අවස්ථා! වීසා සහ ගුවන් ටිකට් නොමිලේ. ලියාපදිංචි ගාස්තුව රු. {AMT} පමණි. අමතන්න {PHONE}",
 ]), ("si", "native", "smishing", "personal", [
 "අම්මා, මගේ ෆෝන් එක කැඩුණා. මේ අලුත් නම්බර් එක. හදිසියේ රු. {AMT}ක් මේ ගිණුමට දාන්න පුලුවන්ද?",
 "ඔයාට වැරදිලා මගේ අංකයට OTP එකක් ආවා. කරුණාකරලා ඒ අංක 6 මට එවන්න.",
 "ඔබගේ WhatsApp ගිණුම වෙනත් දුරකථනයක ලියාපදිංචි වෙමින් පවතී. එය නැවැත්වීමට ඔබට ලැබුණු කේතය එවන්න.",
 "මම පොලිසියෙන්. ඔබගේ නමින් නීති විරෝධී පාර්සලයක් හමුවී ඇත. නඩුවෙන් බේරීමට වහාම {PHONE} අමතන්න.",
 ]), ("si", "native", "spam", "promo", [
 "{TELCO}: මාසයකට රු. {SAMT}ට අසීමිත දත්ත. සක්‍රීය කිරීමට #{OTP}# අමතන්න.",
 "{SHOP} විශේෂ වට්ටම්! තෝරාගත් භාණ්ඩ මත {PCT}% අඩු කිරීම් මෙම සතියේ පමණි.",
 "{SHOP} මහා විකුණුම - {PCT}% දක්වා වට්ටම්. අදම පැමිණෙන්න!",
 "{TELCO}: රු. {SAMT}ට {GB}GB රාත්‍රී දත්ත. ලබා ගැනීමට {SAMT} ලෙස SMS කරන්න.",
 "{SHOP}: පොලී රහිත වාරික ක්‍රමයට ගෘහ උපකරණ. සීමිත කාලයක් පමණි.",
 ]), ("si", "native", "ham", "otp", [
 "ඔබගේ {BANK} OTP අංකය {OTP} වේ. මෙය කිසිවෙකු සමඟ බෙදා නොගන්න.",
 "ඔබගේ ගනුදෙනුව සඳහා එක් වරක් භාවිත මුරපදය {OTP}. වලංගු කාලය මිනිත්තු {MIN}යි. {BANK} කාර්ය මණ්ඩලය කිසිවිට මෙය ඉල්ලන්නේ නැත.",
 "ඔබගේ ගිණුමෙන් ({ACC}) රු. {CENTS}ක් ATM මගින් ආපසු ගන්නා ලදී. ශේෂය රු. {CENTS}.",
 "ඔබගේ ගිණුමට ({ACC}) රු. {CENTS}ක් බැර විය. ස්තූතියි - {BANK}.",
 "ඔබගේ කාඩ්පත මගින් {SHOP} හි රු. {CENTS}ක ගෙවීමක් සිදු විය.",
 ]), ("si", "native", "ham", "service", [
 "ඔබගේ {TELCO} බිල්පත රු. {CENTS} යි. ගෙවීමේ අවසන් දිනය {DATE}.",
 "ඔබගේ විදුලි බිල්පත රු. {CENTS}ක් වේ. ස්තූතියි - ලංවිම.",
 "ඔබගේ {SHOP} ඇණවුම නැව්ගත කර ඇත. ලුහුබැඳීමේ අංකය {TRK}.",
 "{COURIER}: ඔබගේ පාර්සලය අද බෙදාහරිනු ඇත. ලුහුබැඳීමේ අංකය {TRK}.",
 "රු. {SAMT}ක රීලෝඩ් එකක් සාර්ථකව එකතු විය. නව ශේෂය රු. {CENTS}.",
 "ඔබගේ වෛද්‍ය හමුව {DATE} දින {TIME}ට නියම කර ඇත. - {HOSP}",
 "ලංකා තැපෑල: ඔබගේ ලියාපදිංචි ලිපිය {CITY_SI} තැපැල් කාර්යාලයේ ඇත. හැඳුනුම්පත සමඟ පැමිණ ලබාගන්න.",
 ]), ("si", "native", "ham", "personal", [
 "අම්මා මම ගෙදර එන ගමන්. {CITY_SI} ඉඳන් බස් එකේ. පාන් ගෙනියන්න ඕනද?",
 "හෙට ක්ලාස් එක {TIME}ට පටන් ගන්නවා. පරක්කු වෙන්න එපා.",
 "මීටිං එක {TIME}ට කල් දැම්මා. සමාවෙන්න.",
 "බස් එක පරක්කුයි, මම තව විනාඩි {MIN}කින් එනවා.",
 "පන්තියේ ගාස්තුව රු. {SAMT} මේ සතියේ ගෙවන්න අමතක කරන්න එපා.",
 "මම {CITY_SI} ආවා. හවස {TIME}ට හම්බ වෙමු.",
 "ඔයාට එවපු රු. {SAMT} ලැබුණද? කියන්නකෝ.",
 ])]

# ============================ TAMIL ============================
T += [("ta", "native", "smishing", "bank", [
 "{BANK}: உங்கள் கணக்கு தற்காலிகமாக முடக்கப்பட்டுள்ளது. {HRS} மணி நேரத்திற்குள் சரிபார்க்கவும்: {LINK}",
 "அன்புள்ள வாடிக்கையாளரே, உங்கள் {BANK} அட்டை தடுக்கப்பட்டுள்ளது. மீண்டும் செயல்படுத்த OTP எண்ணை இந்த இணைப்பில் உள்ளிடவும் {LINK}",
 "{BANK}: உங்கள் KYC விவரங்கள் புதுப்பிக்கப்படவில்லை. இன்று புதுப்பிக்காவிட்டால் கணக்கு மூடப்படும். {LINK}",
 "உங்கள் கணக்கிலிருந்து ரூ. {AMT} எடுக்கும் கோரிக்கை வந்துள்ளது. இது நீங்கள் இல்லையெனில் உடனே ரத்து செய்யவும்: {LINK}",
 "உங்கள் கிரெடிட் கார்டு வெகுமதி புள்ளிகள் {AMT} இன்று காலாவதியாகும். பணமாக மாற்றவும்: {LINK}",
 "{BANK}: சந்தேகத்திற்குரிய உள்நுழைவு கண்டறியப்பட்டது. கணக்கைப் பாதுகாக்க உங்களுக்கு வந்த குறியீட்டை பதிலாக அனுப்பவும்.",
 "உங்கள் டெபிட் கார்டு காலாவதியாகிவிட்டது. புதிய அட்டைக்கு விவரங்களை உள்ளிடவும்: {LINK}",
 "உங்கள் கடன் விண்ணப்பம் அங்கீகரிக்கப்பட்டது! ரூ. {BAMT} பெற செயலாக்கக் கட்டணம் ரூ. {SAMT} இந்தக் கணக்கில் செலுத்தவும்.",
 "எச்சரிக்கை! உங்கள் {BANK} இணைய வங்கிக் கணக்கில் வேறொரு சாதனத்திலிருந்து உள்நுழைய முயற்சி. உடனே கடவுச்சொல்லை மாற்றவும்: {LINK}",
 "உங்கள் கணக்கில் ரூ. {AMT} தவறுதலாக வரவு வைக்கப்பட்டுள்ளது. திருப்பி அனுப்ப இந்த இணைப்பில் உறுதிப்படுத்தவும் {LINK}",
 ]), ("ta", "native", "smishing", "delivery", [
 "{COURIER}: முகவரி முழுமையாக இல்லாததால் உங்கள் பார்சல் நிறுத்தி வைக்கப்பட்டுள்ளது. ரூ. {SAMT} கட்டணம் செலுத்தி புதுப்பிக்கவும்: {LINK}",
 "உங்கள் பொதி சுங்கத்தில் நிறுத்தப்பட்டுள்ளது. விடுவிக்க சுங்கக் கட்டணம் ரூ. {SAMT} இன்றே செலுத்தவும்: {LINK}",
 "{COURIER}: விநியோகம் தோல்வியடைந்தது. {HRS} மணி நேரத்திற்குள் புதிய தேதியைத் தேர்ந்தெடுக்கவும், இல்லையெனில் பார்சல் திருப்பி அனுப்பப்படும் {LINK}",
 "உங்கள் {SHOP} ஆர்டர் நிறுத்தப்பட்டுள்ளது. கட்டணத்தை உறுதிப்படுத்த அட்டை விவரங்களை உள்ளிடவும்: {LINK}",
 ]), ("ta", "native", "smishing", "government", [
 "போக்குவரத்து அபராதம் ரூ. {SAMT} செலுத்தப்படவில்லை. இன்று செலுத்தாவிட்டால் ஓட்டுநர் உரிமம் இடைநிறுத்தப்படும்: {LINK}",
 "அஸ்வெசும கொடுப்பனவு ரூ. {AMT} உங்கள் பெயரில் அங்கீகரிக்கப்பட்டுள்ளது. பெற வங்கி விவரங்களை உள்ளிடவும்: {LINK}",
 "மின்சார சபை: உங்கள் கட்டணம் செலுத்தப்படாததால் இன்று இரவு மின்சாரம் துண்டிக்கப்படும். உடனே செலுத்தவும் {LINK}",
 "உங்கள் தேசிய அடையாள அட்டை எண்ணை புதுப்பிக்க வேண்டும், இல்லையெனில் SIM முடக்கப்படும். இங்கே புதுப்பிக்கவும்: {LINK}",
 "வருமான வரி திருப்பிச் செலுத்துதல் ரூ. {AMT} உங்களுக்கு உரியது. {HRS} மணி நேரத்திற்குள் விண்ணப்பிக்கவும்: {LINK}",
 ]), ("ta", "native", "smishing", "prize", [
 "வாழ்த்துக்கள்! {TELCO} வெற்றியாளராக ரூ. {BAMT} வென்றுள்ளீர்கள். பரிசைப் பெற {PHONE} அழைக்கவும்.",
 "உங்கள் எண் ஆண்டுவிழா லாட்டரியில் தேர்ந்தெடுக்கப்பட்டுள்ளது! பரிசைப் பெற பதிவுக் கட்டணம் ரூ. {SAMT} {WALLET} செய்யவும்.",
 "{TELCO}: உங்களுக்கு இலவச {GB}GB டேட்டா கிடைத்துள்ளது. இன்றே செயல்படுத்தவும்: {LINK}",
 "{SHOP} பரிசு கூப்பன் ரூ. {AMT} உங்களுக்கு! கணக்கெடுப்பை நிரப்பி பெறுங்கள்: {LINK}",
 ]), ("ta", "native", "smishing", "job", [
 "வீட்டிலிருந்தே வேலை செய்து தினமும் ரூ. {AMT} சம்பாதியுங்கள். WhatsApp: wa.me/{WA}",
 "பகுதிநேர வேலை: வீடியோக்களை like செய்ய தினமும் ரூ. {AMT}. இப்போதே சேருங்கள் {LINK}",
 "வெளிநாட்டு வேலை வாய்ப்புகள்! விசா மற்றும் விமான டிக்கெட் இலவசம். பதிவுக் கட்டணம் ரூ. {AMT} மட்டுமே. அழைக்கவும் {PHONE}",
 ]), ("ta", "native", "smishing", "personal", [
 "அம்மா, என் போன் உடைந்துவிட்டது. இது என் புதிய எண். அவசரமாக ரூ. {AMT} இந்தக் கணக்கில் போட முடியுமா?",
 "தவறுதலாக உங்கள் OTP என் எண்ணுக்கு வந்துவிட்டது. தயவுசெய்து அந்த 6 இலக்கங்களை எனக்கு அனுப்புங்கள்.",
 "உங்கள் WhatsApp கணக்கு வேறொரு போனில் பதிவு செய்யப்படுகிறது. அதை நிறுத்த உங்களுக்கு வந்த குறியீட்டை அனுப்புங்கள்.",
 "நான் காவல்துறையிலிருந்து பேசுகிறேன். உங்கள் பெயரில் சட்டவிரோத பார்சல் கண்டுபிடிக்கப்பட்டுள்ளது. உடனே {PHONE} அழைக்கவும்.",
 ]), ("ta", "native", "spam", "promo", [
 "{TELCO}: மாதம் ரூ. {SAMT}க்கு வரம்பற்ற டேட்டா. செயல்படுத்த #{OTP}# அழைக்கவும்.",
 "{SHOP} சிறப்பு தள்ளுபடி! தேர்ந்தெடுக்கப்பட்ட பொருட்களுக்கு {PCT}% குறைப்பு இந்த வாரம் மட்டும்.",
 "{SHOP} பெரிய விற்பனை - {PCT}% வரை தள்ளுபடி. இன்றே வாருங்கள்!",
 "{TELCO}: ரூ. {SAMT}க்கு {GB}GB இரவு டேட்டா. பெற {SAMT} என SMS செய்யவும்.",
 "{SHOP}: வட்டியில்லா தவணையில் வீட்டு உபகரணங்கள். குறுகிய காலம் மட்டும்.",
 ]), ("ta", "native", "ham", "otp", [
 "உங்கள் {BANK} OTP எண் {OTP}. இதை யாருடனும் பகிர வேண்டாம்.",
 "உங்கள் பரிவர்த்தனைக்கான ஒருமுறை கடவுச்சொல் {OTP}. {MIN} நிமிடங்களுக்கு செல்லுபடியாகும். {BANK} ஊழியர்கள் இதை ஒருபோதும் கேட்கமாட்டார்கள்.",
 "உங்கள் கணக்கிலிருந்து ({ACC}) ATM மூலம் ரூ. {CENTS} எடுக்கப்பட்டது. மீதி ரூ. {CENTS}.",
 "உங்கள் கணக்கில் ({ACC}) ரூ. {CENTS} வரவு வைக்கப்பட்டுள்ளது. நன்றி - {BANK}.",
 "உங்கள் அட்டை மூலம் {SHOP} இல் ரூ. {CENTS} செலுத்தப்பட்டது.",
 ]), ("ta", "native", "ham", "service", [
 "உங்கள் {TELCO} கட்டணம் ரூ. {CENTS}. செலுத்த வேண்டிய இறுதி நாள் {DATE}.",
 "உங்கள் மின்சாரக் கட்டணம் ரூ. {CENTS}. நன்றி - இலங்கை மின்சார சபை.",
 "உங்கள் {SHOP} ஆர்டர் அனுப்பப்பட்டது. கண்காணிப்பு எண் {TRK}.",
 "{COURIER}: உங்கள் பார்சல் இன்று விநியோகிக்கப்படும். கண்காணிப்பு எண் {TRK}.",
 "ரூ. {SAMT} ரீலோட் வெற்றிகரமாக சேர்க்கப்பட்டது. புதிய மீதி ரூ. {CENTS}.",
 "உங்கள் மருத்துவ சந்திப்பு {DATE} அன்று {TIME} மணிக்கு நிர்ணயிக்கப்பட்டுள்ளது. - {HOSP}",
 "இலங்கை தபால்: உங்கள் பதிவுத் தபால் {CITY_TA} தபால் நிலையத்தில் உள்ளது. அடையாள அட்டையுடன் வந்து பெற்றுக்கொள்ளவும்.",
 ]), ("ta", "native", "ham", "personal", [
 "அம்மா நான் வீட்டுக்கு வந்துகொண்டிருக்கிறேன். {CITY_TA} பஸ்ஸில் இருக்கிறேன். பாண் வாங்கி வரவா?",
 "நாளை வகுப்பு {TIME} மணிக்கு தொடங்கும். தாமதிக்க வேண்டாம்.",
 "கூட்டம் {TIME} மணிக்கு மாற்றப்பட்டது. மன்னிக்கவும்.",
 "பஸ் தாமதம், இன்னும் {MIN} நிமிடத்தில் வருகிறேன்.",
 "டியூஷன் கட்டணம் ரூ. {SAMT} இந்த வாரம் செலுத்த மறக்காதே.",
 "நான் {CITY_TA} வந்துவிட்டேன். மாலை {TIME} மணிக்கு சந்திப்போம்.",
 "உனக்கு அனுப்பிய ரூ. {SAMT} கிடைத்ததா? சொல்லு.",
 ])]

# ============================ SINGLISH ============================
T += [("si-en", "roman", "smishing", "bank", [
 "{BANK}: oyage account eka temporary block karala. Paya {HRS}k athulatha verify karanna: {LINK}",
 "Dear customer oyage {BANK} card eka block wela. Aye activate karanna OTP eka me link ekata danna {LINK}",
 "{BANK}: oyage KYC update karala na. Ada update kale nattam account eka close wenawa. {LINK}",
 "Oyage account eken Rs.{AMT}k ganna request ekak awa. Eka oya nemei nam dan cancel karanna: {LINK}",
 "Credit card points {AMT} ada expire wenawa. Salli walata convert karaganna: {LINK}",
 "Loan eka approve una! Rs.{BAMT} ganna processing fee Rs.{SAMT} me account ekata danna.",
 "{BANK}: suspicious login ekak detect una. Account eka safe karanna oyata awa code eka reply karanna.",
 "Warning! Oyage {BANK} online banking ekata wena device ekakin log wenna try kala. Dan ma password eka change karanna: {LINK}",
 ]), ("si-en", "roman", "smishing", "delivery", [
 "{COURIER}: oyage parcel eka address eka waradi nisa hold karala. Rs.{SAMT} fee eka pay karala update karanna: {LINK}",
 "Oyage parcel eka customs eke hira wela. Release karanna Rs.{SAMT} ada gewanna: {LINK}",
 "Oyage {SHOP} order eka hold wela. Payment confirm karanna card details danna: {LINK}",
 "{COURIER} delivery eka fail una. Paya {HRS}k athulatha aluth date ekak select karanna nattam parcel eka return karanawa {LINK}",
 ]), ("si-en", "roman", "smishing", "government", [
 "Traffic fine Rs.{SAMT} gewala na. Ada gewwe nattam license eka suspend karanawa: {LINK}",
 "Aswesuma Rs.{AMT} oyage namata approve wela. Ganna bank details danna: {LINK}",
 "CEB: bill eka gewala nathi nisa ada raa current eka cut karanawa. Dan ma pay karanna {LINK}",
 "NIC number eka update karanna ona, nattam SIM eka deactivate wenawa: {LINK}",
 "Tax refund Rs.{AMT} oyata thiyenawa. Paya {HRS}k athulatha claim karanna: {LINK}",
 ]), ("si-en", "roman", "smishing", "prize", [
 "Congratulations! Oya {TELCO} lucky draw eken Rs.{BAMT} dinala. Prize eka ganna {PHONE} ta call karanna.",
 "Oyage number eka anniversary lottery ekata select wela! Registration fee Rs.{SAMT} {WALLET} karanna.",
 "{TELCO} eken free {GB}GB data awa. Ada ma activate karanna: {LINK}",
 "{SHOP} gift voucher Rs.{AMT} oyata! Survey eka fill karala ganna: {LINK}",
 ]), ("si-en", "roman", "smishing", "job", [
 "Gedara idan wada karala dawasakata Rs.{AMT} hoyanna. WhatsApp karanna wa.me/{WA}",
 "Part time job: videos like karanna dawasata Rs.{AMT}. Dan ma join wenna {LINK}",
 "Foreign job! Visa ticket free. Registration Rs.{AMT} witharai. Call {PHONE}",
 ]), ("si-en", "roman", "smishing", "personal", [
 "Amma mage phone eka kaduna. Me aluth number eka. Hadisiyen Rs.{AMT}k me account ekata danna puluwanda?",
 "Wardila oyage OTP eka mage number ekata awa. Please eka mata ewanna.",
 "Oyage WhatsApp eka wena phone ekaka register wenawa. Eka nawaththanna oyata awa code eka ewanna.",
 "Machan urgent Rs.{SAMT}k ona, me number ekata {WALLET} karanna. Heta denawa sure.",
 ]), ("si-en", "roman", "spam", "promo", [
 "{TELCO}: masayata Rs.{SAMT} ta unlimited data. Activate karanna #{OTP}# dial karanna.",
 "{SHOP} special offer! Select karapu items walata {PCT}% off me sathiye witharai.",
 "{SHOP} big sale - {PCT}% dakwa off. Ada ma enna!",
 "{TELCO}: Rs.{SAMT} ta {GB}GB night data. {SAMT} kiyala SMS karanna.",
 ]), ("si-en", "roman", "ham", "otp", [
 "Your {BANK} OTP is {OTP}. Kisiwekutath share karanna epa.",
 "Rs.{CENTS} withdrawn from A/C {ACC} via ATM. Bal Rs.{CENTS}",
 "Rs.{CENTS} credited to A/C {ACC}. Thank you - {BANK}",
 "{OTP} is your {SHOP} verification code. Valid for {MIN} minutes.",
 ]), ("si-en", "roman", "ham", "service", [
 "Oyage {TELCO} bill eka Rs.{CENTS}. Last date {DATE}.",
 "{SHOP} order eka ship kala. Tracking no {TRK}.",
 "Rs.{SAMT} reload eka success. Aluth balance Rs.{CENTS}",
 "Oyage doctor appointment eka {DATE} {TIME} ta. - {HOSP}",
 "{COURIER}: oyage parcel eka ada deliver karanawa. Tracking {TRK}.",
 ]), ("si-en", "roman", "ham", "personal", [
 "Amma mama gedara enawa, {CITY} idan bus eke. Paan ganna onada?",
 "Heta class eka {TIME} ta. Late wenna epa.",
 "Meeting eka {TIME} ta damma. Sorry.",
 "Bus eka late, thawa minute {MIN}kin ennam.",
 "Tuition fee Rs.{SAMT} me sathiye gewanna amathaka karanna epa.",
 "Mama {CITY} awa. Hawasa {TIME} ta hambawemu.",
 "Machan match eka balanna yanawada? Mama ticket gaththa Rs.{SAMT} ta.",
 "Ewapu Rs.{SAMT} labuna da? Kiyanna.",
 ])]

# ============================ TANGLISH ============================
T += [("ta-en", "roman", "smishing", "bank", [
 "{BANK}: unga account temporary-a block pannirukkom. {HRS} mani neraththukkul verify pannunga: {LINK}",
 "Dear customer unga {BANK} card block aayirukku. Activate panna OTP-a indha link-la podunga {LINK}",
 "{BANK}: unga KYC update pannala. Inniku update pannalana account close aagum. {LINK}",
 "Unga account-la irundhu Rs.{AMT} edukka request vandhirukku. Adhu neenga illana ippove cancel pannunga: {LINK}",
 "Credit card points {AMT} inniku expire aagum. Panama maathunga: {LINK}",
 "Loan approve aayiduchu! Rs.{BAMT} vaanga processing fee Rs.{SAMT} indha account-ku podunga.",
 "{BANK}: suspicious login kandupidikkapattadhu. Account-a safe panna unga-ku vandha code-a reply pannunga.",
 "Warning! Unga {BANK} online banking-la vera device-la irundhu login try pannirukkaanga. Udane password maathunga: {LINK}",
 ]), ("ta-en", "roman", "smishing", "delivery", [
 "{COURIER}: address sariyaa illadhadhaala unga parcel hold pannirukkom. Rs.{SAMT} fee kattunga: {LINK}",
 "Unga parcel customs-la maatikittu irukku. Release panna Rs.{SAMT} inniku kattunga: {LINK}",
 "Unga {SHOP} order hold-la irukku. Payment confirm panna card details podunga: {LINK}",
 "{COURIER} delivery fail aayiduchu. {HRS} mani neraththukkul pudhu date select pannunga illana parcel return aagum {LINK}",
 ]), ("ta-en", "roman", "smishing", "government", [
 "Traffic fine Rs.{SAMT} kattala. Inniku kattalana license suspend pannuvom: {LINK}",
 "Aswesuma Rs.{AMT} unga perla approve aayirukku. Vaanga bank details podunga: {LINK}",
 "CEB: bill kattaadhadhaala inniku raathiri current cut pannuvom. Udane kattunga {LINK}",
 "NIC number update pannanum, illana SIM deactivate aagum: {LINK}",
 "Tax refund Rs.{AMT} ungalukku irukku. {HRS} mani neraththukkul claim pannunga: {LINK}",
 ]), ("ta-en", "roman", "smishing", "prize", [
 "Congratulations! Neenga {TELCO} lucky draw-la Rs.{BAMT} jeyichirukeenga. Prize vaanga {PHONE} ku call pannunga.",
 "Unga number anniversary lottery-la select aayirukku! Registration fee Rs.{SAMT} {WALLET} pannunga.",
 "{TELCO}-la irundhu free {GB}GB data vandhirukku. Inniku activate pannunga: {LINK}",
 "{SHOP} gift voucher Rs.{AMT} ungalukku! Survey fill panni vaanguga: {LINK}",
 ]), ("ta-en", "roman", "smishing", "job", [
 "Veetla irundhe vela senju dhinamum Rs.{AMT} sambaadhinga. WhatsApp pannunga wa.me/{WA}",
 "Part time job: videos like panna dhinam Rs.{AMT}. Ippove join pannunga {LINK}",
 "Foreign job! Visa ticket free. Registration Rs.{AMT} mattum. Call {PHONE}",
 ]), ("ta-en", "roman", "smishing", "personal", [
 "Amma en phone odanjiduchu. Idhu en pudhu number. Avasarama Rs.{AMT} indha account-ku poda mudiyuma?",
 "Thavaraa unga OTP en number-ku vandhuduchu. Please adha enakku anuppunga.",
 "Unga WhatsApp vera phone-la register aagudhu. Adha niruththa unga-ku vandha code-a anuppunga.",
 "Machan urgent-a Rs.{SAMT} venum, indha number-ku {WALLET} pannu. Naalaikku tharen sure.",
 ]), ("ta-en", "roman", "spam", "promo", [
 "{TELCO}: maasam Rs.{SAMT} ku unlimited data. Activate panna #{OTP}# dial pannunga.",
 "{SHOP} special offer! Select panna items-ku {PCT}% off indha vaaram mattum.",
 "{SHOP} big sale - {PCT}% varai off. Inniku vaanga!",
 "{TELCO}: Rs.{SAMT} ku {GB}GB night data. {SAMT} nu SMS pannunga.",
 ]), ("ta-en", "roman", "ham", "otp", [
 "Your {BANK} OTP is {OTP}. Yaarkitayum share panna vendaam.",
 "Rs.{CENTS} withdrawn from A/C {ACC} via ATM. Bal Rs.{CENTS}.",
 "Rs.{CENTS} credited to A/C {ACC}. Thank you - {BANK}.",
 "{OTP} is your {SHOP} verification code. Valid for {MIN} minutes.",
 ]), ("ta-en", "roman", "ham", "service", [
 "Unga {TELCO} bill Rs.{CENTS}. Last date {DATE}.",
 "{SHOP} order ship aayiduchu. Tracking no {TRK}.",
 "Rs.{SAMT} reload success. Pudhu balance Rs.{CENTS}.",
 "Unga doctor appointment {DATE} {TIME} ku. - {HOSP}",
 "{COURIER}: unga parcel inniku deliver aagum. Tracking {TRK}.",
 ]), ("ta-en", "roman", "ham", "personal", [
 "Amma naan veetukku vandhukittu irukken, {CITY} bus-la. Paan vaanganuma?",
 "Naalaikku class {TIME} manikku. Late aagaadhe.",
 "Meeting {TIME} manikku maathiyaachu. Sorry.",
 "Bus late, innum {MIN} nimishathula varen.",
 "Tuition fee Rs.{SAMT} indha vaaram katta marakkaadhe.",
 "Naan {CITY} vandhuten. Saayangaalam {TIME} ku meet pannalaam.",
 "Machan match paakka poriya? Naan ticket vaanginen Rs.{SAMT}-ku.",
 "Anuppuna Rs.{SAMT} kedachudha? Sollu.",
 ])]


def main() -> None:
    rows, seen = [], set()
    for lang, script, label, cat, temps in T:
        for i, t in enumerate(temps):
            tid = f"{lang}-{label}-{cat}-{i:02d}"
            has_slot = "{" in t
            n = VARIANTS if has_slot else 1
            tries = 0
            made = 0
            while made < n and tries < n * 5:
                tries += 1
                txt = fill(t)
                if txt in seen:
                    continue
                seen.add(txt)
                rows.append(dict(text=txt, label=label, language=lang, script=script,
                                 category=cat, origin="template", template_id=tid))
                made += 1

    # keep the original hand-written one-off messages too (each is its own group)
    hand = ROOT / "data" / "raw" / "multilingual_handwritten.csv"
    if hand.exists():
        with open(hand, encoding="utf-8") as f:
            for j, r in enumerate(csv.DictReader(f)):
                if r["text"] not in seen:
                    seen.add(r["text"])
                    rows.append(dict(text=r["text"], label=r["label"], language=r["language"], script=r.get("script", ""),
                                     category=r.get("category", ""), origin="handwritten", template_id=f"hand-{j:03d}"))

    rng.shuffle(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["text", "label", "language", "script", "category", "origin", "template_id"])
        w.writeheader()
        w.writerows(rows)

    from collections import Counter
    c = Counter((r["language"], r["label"]) for r in rows)
    print(f"wrote {OUT}  rows={len(rows)}  templates={len({r['template_id'] for r in rows})}")
    for lang in ["si", "ta", "si-en", "ta-en"]:
        print(f"  {lang:<6}", {k: c[(lang, k)] for k in ["smishing", "spam", "ham"]})


if __name__ == "__main__":
    main()
