import os, re, sys, zipfile, tempfile, requests
sys.modules["torch.classes"] = None

import fitz, torch, pytesseract, pandas as pd
from PIL import Image
from io import BytesIO
import streamlit as st
from textwrap import wrap
from fuzzywuzzy import fuzz
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import torch.nn.functional as F
from urllib.parse import urljoin
from torchvision import transforms
from type import detect_product_type
import matplotlib.backends.backend_pdf
from name import apollo
from pdf2image import convert_from_path
from torchvision.models import resnet50, ResNet50_Weights
from sentence_transformers import SentenceTransformer, util

# Setup Streamlit
st.set_page_config(page_title="Valve Data Integrity Checker", page_icon="🛠️", layout="wide")
st.markdown('<style>.main{background:#f9f9f9}.stButton>button{background:#add8e6;color:#000}.stTextInput input{background:#fff;color:#00040D;caret-color: black;}</style>', unsafe_allow_html=True)

# Initialize session state
defaults = dict(crop_left=5, crop_top=10, crop_right=60, crop_bottom=45, validation_started=False, slider_changed=False)
for k, v in defaults.items(): st.session_state.setdefault(k, v)

def slider_callback(): st.session_state.slider_changed = True

# Size Detection Functions
def extract_size_values_from_website(soup):
    sizes = [re.search(r'(\d[\d\s/-]*\")', o.text).group(1).strip() for o in soup.select('#childProudcts option') if '"' in o.text and re.search(r'(\d[\d\s/-]*\")', o.text)]
    return list(set(sizes))

def extract_sizes_from_pdf(text):
    raw = re.findall(r'\b\d[\d\s/-]*(\"| inch|in\b)', text.lower())
    return list(set(s.replace('inch', '"').replace('in', '"').strip() for s in raw))

def normalize_size(s):
    s = s.replace(' ', '').replace('inch', '').replace('in', '').replace('"', '')
    if '-' in s:
        parts = s.split('-')
        try: return str(float(parts[0]) + float(parts[1])/12)
        except: return s
    elif '/' in s:
        try: num, den = s.split('/'); return str(round(float(num)/float(den), 4))
        except: return s
    return s

def extract_product_benefits_or_features(text):
    features = []
    lines = text.splitlines()

    capture = False
    max_lines = 25
    captured_lines = 0

    for i, line in enumerate(lines):
        line_stripped = line.strip()

        # Detect start of FEATURES section
        if not capture and re.search(r'\bfeatures\b', line_stripped, re.IGNORECASE):
            capture = True
            continue

        # Stop at next known section header
        if capture and any(
            re.match(rf"^{s}[:\s]*$", line_stripped, re.IGNORECASE)
            for s in ["options", "performance rating", "standard materials", "precautionary note", "description"]
        ):
            break

        if capture:
            cleaned = line_stripped.strip("•*- \u2022").strip()
            if 3 < len(cleaned) < 150:
                features.append(cleaned.capitalize())
            captured_lines += 1
            if captured_lines > max_lines:
                break

    # Fallback: inline FEATURES section (e.g., "features • item1 • item2 ...")
    if not features:
        inline_match = re.search(
            r'features\s*[:\-]?\s*(.+?)(options|performance rating|standard materials|description|precautionary note|see installation manual)',
            text,
            re.IGNORECASE | re.DOTALL
        )
        if inline_match:
            feature_blob = inline_match.group(1)
            parts = re.split(r'•|\n| - |\s{2,}', feature_blob)
            for item in parts:
                item_clean = item.strip("•*- \u2022").strip()
                if 3 < len(item_clean) < 150:
                    features.append(item_clean.capitalize())

    return [("Features", features)] if features else []


def render_benefits_and_features_section(full_pdf_text):
    benefit_feature_sections = extract_product_benefits_or_features(full_pdf_text)

    if benefit_feature_sections:
        for section_name, items in benefit_feature_sections:
            st.markdown(f"### 💡 Product {section_name}")
            for item in items:
                st.write(f"- {item}")
    else:
        st.markdown("### 💡 Product Features")
        st.info("No specific product features section found in the spec sheet.")

def find_best_pdf_link(soup, base_url):
    links = [a for a in soup.find_all("a", href=True) if ".pdf" in a['href'].lower()]
    for keyword in ["specif", "specification", "manual", "catalog"]:
        for tag in links:
            if tag.string and keyword in tag.string.lower():
                return urljoin(base_url, tag['href'])  # fix here
    return urljoin(base_url, links[0]['href']) if links else None

@st.cache_resource
def load_embedder():
    return SentenceTransformer('all-MiniLM-L6-v2')

embedder = load_embedder()

def generate_action_required_semantic(expected_value, pdf_text):
    if not expected_value or not pdf_text:
        return "Unable to reason – missing input"
    try:
        snippets = pdf_text.split('.')
        sims = [(s.strip(), util.cos_sim(embedder.encode(expected_value), embedder.encode(s.strip())).item())
                for s in snippets if len(s.strip()) > 10]
        if not sims:
            return "Mismatch – No context found in PDF related to expected value"
        top_snip, top_score = max(sims, key=lambda x: x[1])
        top_snip_clean = top_snip.replace("\n", " ").strip()
        if top_score > 0.85:
            return f"No action required – Clearly supported in PDF"
        if top_score > 0.65:
            if any(keyword in top_snip_clean.lower() for keyword in ["range", "limit", "between", "min", "max"]):
                return f"Review range interpretation – Value may be part of a conditional or bounded spec: '{top_snip_clean[:120]}...'"
            elif any(unit in expected_value.lower() for unit in ["psi", "°", "%", "mm", "bar"]):
                return f"Confirm units – Potential unit mismatch or formatting issue: '{top_snip_clean[:120]}...'"
            else:
                return f"Review phrasing – Semantics partially aligned but format differs: '{top_snip_clean[:120]}...'"
        if top_score > 0.4:
            if len(expected_value) > 30:
                return f"Possible layout or OCR formatting issue – Similar phrase found but weak correlation: '{top_snip_clean[:120]}...'"
            return f"Mismatch – Similar word forms found but value context unclear: '{top_snip_clean[:120]}...'"
        return f"Mismatch – No reliable semantic match. Closest found: '{top_snip_clean[:120]}...'"
    except Exception as e:
        return f"Semantic reasoning failed: {str(e)}"

# Utility functions
def sanitize_filename(name): return re.sub(r'[\\/*?:"<>|]', "", name)
@st.cache_data(show_spinner=False)
def extract_text_from_image_pdf(pdf_path): return " ".join(pytesseract.image_to_string(img) for img in convert_from_path(pdf_path, dpi=300))
@st.cache_resource
def load_resnet(): image_model = resnet50(weights=ResNet50_Weights.DEFAULT); image_model.eval(); return image_model

@st.cache_data(show_spinner=False)
def extract_critical_info_warnings_only(text):
    warning_patterns = [
        r'precautionary note[:\-]?\s*(.+?[\.\!](?=\s+[A-Z]|$))',
        r'\*+\s*lead free[:\-]?\s*(.+?[\.\!](?=\s+[A-Z]|$))',
        r'(not recommended[^\.!\n]{10,200}[\.\!])',
        r'(the wetted surfaces of this product shall contain no more than[^\.!\n]{10,200}[\.\!])',
    ]

    seen = set()
    results = []

    for pattern in warning_patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        for match in matches:
            sentence = match.strip()

            # Ensure only reasonable length and avoid duplicates
            if 20 < len(sentence) < 300:
                normalized = sentence[0].upper() + sentence[1:]
                if normalized.lower() not in seen:
                    seen.add(normalized.lower())
                    results.append(f"- {normalized}")

    return "\n" + "\n".join(results) if results else "No warnings detected."

def keyword_highlights_smart(text, keyword_dict):
    results = []
    lines = text.split('\n')

    # Collect candidate "structured" lines
    table_lines = []
    for line in lines:
        clean = line.strip()
        if 20 < len(clean) < 180 and (
            clean.count('  ') >= 1 or '\t' in clean or ' | ' in clean or re.search(r"\d+\s*[a-zA-Z%°]", clean)
        ):
            table_lines.append(clean)

    table_text = '\n'.join(table_lines)
    fallback_used = False

    for main_kw, variants in keyword_dict.items():
        found_snippets = []

        # Primary: Try matching only in table-like text
        for variant in variants:
            pattern = rf'(.{{0,40}}{re.escape(variant).replace(r"\.", r"[\.\s]?")}.{{0,40}})'
            for match in re.finditer(pattern, table_text, re.IGNORECASE):
                snippet = match.group(0).strip().replace('\n', ' ')
                found_snippets.append(f"...{snippet}...")

        # Fallback: if nothing found, try full text
        if not found_snippets:
            fallback_used = True
            for variant in variants:
                for match in re.finditer(rf'(.{{0,40}}{re.escape(variant)}.{{0,40}})', text, re.IGNORECASE):
                    snippet = match.group(0).strip().replace('\n', ' ')
                    if snippet not in found_snippets:
                        found_snippets.append(f"...{snippet}...")
                    if len(found_snippets) >= 2:
                        break

        if found_snippets:
            wrapped_snippet = '\n'.join(wrap("; ".join(found_snippets), 50))
            results.append({"Keyword": main_kw, "Extracted Data": wrapped_snippet})

    if not results:
        results = [{"Keyword": "None", "Extracted Data": "No keywords matched."}]
    return results


# Improved PDF image extraction: prioritize large, wide product images
def extract_best_pdf_image(doc):
    best_image = None
    best_score = 0

    for page in doc:
        for img in page.get_images(full=True):
            try:
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                img_pil = Image.open(BytesIO(image_bytes)).convert("RGB")
                width, height = img_pil.size
                area = width * height
                aspect_ratio = width / height if height != 0 else 1
                score = area * (1.2 if aspect_ratio > 1 else 1.0)

                if width < 100 or height < 100:
                    continue  # skip icons and logos

                if score > best_score:
                    best_image = img_pil
                    best_score = score
            except:
                continue

    return best_image

model = load_resnet()
preprocess = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

if "validation_started" not in st.session_state:
    st.session_state.validation_started = False

# UI Inputs
st.title("🛠️ Valve Data Integrity Checker")

# CSV or URL Option
st.markdown("<h4>📂 You may input a URL or upload a CSV with 'URL' column.</h4>", unsafe_allow_html=True)
url = st.text_input("Product URL")
uploaded_csv = st.file_uploader("Or upload CSV with 'URL' column", type=["csv"])

urls = []
if uploaded_csv is not None:
    try:
        df = pd.read_csv(uploaded_csv)
        if 'URL' not in df.columns:
            st.warning("Uploaded CSV must have a 'URL' column.")
            st.stop()

        # ✅ Add Brand filter if 'Brand Name' column exists
        if 'Brand Name' in df.columns:
            unique_brands = df['Brand Name'].dropna().unique().tolist()
            selected_brands = st.multiselect("🧯 Filter by Brand Name", options=unique_brands, default=unique_brands)

            # Filter the dataframe based on selected brands
            df = df[df['Brand Name'].isin(selected_brands)]

        urls = df['URL'].dropna().tolist()

    except Exception as e:
        st.error(f"Error reading uploaded CSV: {e}")
        st.stop()
elif url:
    urls = [url]

# Global user keyword input (applies to all URLs in current session)
default_keywords = ""
global_keyword_input = st.text_input("Keywords to look out for (comma-separated)", value=default_keywords, key="global_keyword_input")
global_keywords = [kw.strip().lower() for kw in global_keyword_input.split(',') if kw.strip()]

# Button Logic
button_1, button_2 = st.columns([1,2])
with button_1:
    validate_clicked, reset_clicked = st.columns([1,2])
    validate_clicked = validate_clicked.button("Validate Specifications")
    if not validate_clicked and not st.session_state.get("validation_started"):
        st.stop()
    reset_clicked = reset_clicked.button("Reset Session")

if reset_clicked:
    st.session_state.clear()
    st.stop()

if (validate_clicked or st.session_state.get("validation_rerun")) and not st.session_state.get("slider_changed", False):
    st.session_state.validation_started = True
    st.session_state.validation_rerun = False
    st.session_state.slider_changed = False
    st.write("Validation logic executing...")

def normalize_string(s):
    s = re.sub(r'["“”]', ' inch', s.lower().replace('to', '-'))
    s = re.sub(r'(\s*°\s*|\s*f\b|degree)', 'f', s)
    s = re.sub(r'\s+|-|[^a-z0-9]', '', s)
    return s

def synonym_replace(s):
    synonyms = {
        "lf brass": "lead free brass", "lead-free brass": "lead free brass", "c28500": "lead free brass",
        "slt. coated": "", "slt coated": "", "slt": "", "(slt. coated)": "", "(slt coated)": ""
    }
    s = s.lower()
    for key, val in synonyms.items():
        s = s.replace(key, val)
    return s.strip()

def engineering_equiv_normalize(text):
    text = text.lower().replace("minimum", "min.").replace("maximum", "max.")
    text = re.sub(r"\s+", " ", text.replace("–", "-").replace("°", " degrees ")).strip()
    return text
    
try:
    if not urls:
        st.warning("Please enter at least one product URL.")
        st.stop()

    for product_url in urls:
        response = requests.get(product_url)
        soup = BeautifulSoup(response.text, 'html.parser')

        title_element = soup.find("h1", class_="productView-title")
        if title_element:
            full_title = title_element.get_text(strip=True)
            readable_name = full_title.strip()
        else:
            # Fallback to URL-based name
            name_segment = product_url.split("https://valveman.com/products/")[-1].split("-")[:10]
            readable_name = " ".join(name_segment).strip()
        st.subheader(f"Product Name: {readable_name}")
        pdf_url = find_best_pdf_link(soup, product_url)
        if not pdf_url:
            st.warning(f"No specification PDF found for {product_url}")
            st.markdown("<hr style='border: 2px solid #bbb;'>", unsafe_allow_html=True)
            continue
        pdf_links_text = []

        # Display product page link
        st.markdown(f'<a href="{product_url}" target="_blank" style="text-decoration:none;font-weight:600;">🔗 Product Page Link</a>', unsafe_allow_html=True)

        # Detect and display all relevant PDF links horizontally
        pdf_links = soup.find_all("a", href=True)
        pdf_labels_map = {
            "quick start guide": "Quick Start Guide", 
            "instruction" : "Instruction Manual",
            "specification": "Specification Sheet",
            "specif": "Specification Sheet",
            "certificate": "Certificate",
            "datasheet": "Datasheet",
            "brochure": "Brochure",
            "catalog": "Catalog",
            "manual": "Manual",
        }
        displayed_links = []

        for tag in pdf_links:
            href = tag['href']
            if ".pdf" in href.lower():
                label = "PDF"
                for keyword, nice_label in pdf_labels_map.items():
                    if keyword in tag.text.lower() or keyword in href.lower():
                        label = nice_label
                        break
                displayed_links.append((label, href))

        if displayed_links:
            cols = st.columns(len(displayed_links))
            for i, (label, link) in enumerate(displayed_links):
                with cols[i]:
                    st.markdown(f'<a href="{link}" target="_blank" style="text-decoration:none;font-weight:600;">📄 {label}</a>', unsafe_allow_html=True)

        else:
            st.warning("No PDF links found for this product.")

        # Extract main product image URL
        image_url = None
        img_tag = soup.select_one("section.productView-images figure img")
        if img_tag and img_tag.has_attr("src"):
            image_url = img_tag["src"]

        # Extract technical specs
        description = soup.select_one('div[data-tab="description"]') or soup
        spec_rows = description.select('div.tech .row')
        fields_to_check = []
        website_sizes = []
        for row in spec_rows:
            label = row.select_one('.left')
            value = row.select_one('.right')
            if label and value:
                fields_to_check.append((label.text.strip(), value.text.strip()))
                if 'size' in label.text.strip().lower():
                    website_sizes.append(value.text.strip())

        if not pdf_url:
            st.warning(f"No specification PDF found for {product_url}")
            continue

        temp_dir = tempfile.gettempdir()
        safe_filename = re.sub(r'[\\:*?<>|]', "", readable_name).replace('"', 'in').replace('/', '-')
        pdf_path = os.path.join(tempfile.gettempdir(), f"{safe_filename}_spec.pdf")


        # Download the PDF
        pdf_content = requests.get(pdf_url).content
        with open(pdf_path, 'wb') as f:
            f.write(pdf_content)

        # try:
        #     st.write(f"📥 Downloading PDF: {pdf_url}")
        #     response = requests.get(pdf_url, timeout=20)  # Set timeout to 20 seconds
        #     pdf_content = response.content

        #     with open(pdf_path, 'wb') as f:
        #         f.write(pdf_content)

        # except Exception as e:
        #     st.error(f"⏳ Failed to fetch PDF for {readable_name}: {e}")
        #     st.markdown("<hr style='border: 1px solid #ccc;'>", unsafe_allow_html=True)
        #     continue

        # Read PDF text
        doc = fitz.open(pdf_path)
        page_sources = [(i, p.get_text()) for i, p in enumerate(doc)]
        all_pdf_text = [text for _, text in page_sources]

        if not any(all_pdf_text) or all(len(text.strip()) < 20 for text in all_pdf_text):
            st.info(f"OCR fallback activated for scanned or unreadable PDF...")
            ocr_text = extract_text_from_image_pdf(pdf_path)
            all_pdf_text = [ocr_text]
            page_sources = [(0, ocr_text)]

        full_pdf_text = ' '.join(all_pdf_text).lower()
        full_pdf_text = re.sub(r'\s+', ' ', full_pdf_text)

        keywords_to_look_for = global_keywords

        website_size_labels = extract_size_values_from_website(soup)
        pdf_size_labels = extract_sizes_from_pdf(full_pdf_text)
        
        website_sizes_norm = set(normalize_size(s) for s in website_size_labels)
        pdf_sizes_norm = set(normalize_size(s) for s in pdf_size_labels)
        
        missing_on_website = pdf_sizes_norm - website_sizes_norm
        missing_in_pdf = website_sizes_norm - pdf_sizes_norm

        detected_valve_type = detect_product_type(readable_name)
        st.markdown(f"### Product Type: **_{detected_valve_type}_**")

        standard_name = apollo(readable_name, full_pdf_text)
        st.markdown(f"### Proposed Product Name: **_{standard_name}_**")
        
        # Compare specifications
        results = []
        total_matches = 0
        spelling_variants = {
            "aluminum": "aluminium", "aluminium": "aluminum", "center": "centre", "colour": "color", "meter": "metre",
        }
        for label, expected_value in fields_to_check:
            label_clean = label.strip().lower()
            value_clean = expected_value.strip().lower()

            if label_clean in ["approvals", "rohs", "specs designed to"]:
                continue
            expected_value_clean = re.sub(r'\s+', ' ', expected_value.strip().lower())
            normalized_expected = normalize_string(synonym_replace(expected_value_clean))
            normalized_pdf = normalize_string(synonym_replace(full_pdf_text))
            exact_match = expected_value_clean in full_pdf_text
            normalized_match = normalized_expected in normalized_pdf
            range_match = False
            range_partial_match = False
            range_pattern = re.match(r"(\d+)\s*(to|\-|\u2013|\u2014)\s*(\d+)(\s*psi)?", expected_value_clean)
            if range_pattern:
                start_psi = int(range_pattern.group(1))
                end_psi = int(range_pattern.group(3))
                if start_psi == end_psi:
                    required_range = [start_psi]
                else:
                    step = 10 if (end_psi - start_psi) % 10 == 0 else 5
                    required_range = list(range(start_psi, end_psi + 1, step))

                possible_numbers = re.findall(r'\b([\dOIlS]{2,4})\b', full_pdf_text)
                cleaned_numbers = []
                for val in possible_numbers:
                    cleaned = val.upper().replace('O', '0').replace('I', '1').replace('L', '1').replace('S', '5')
                    try:
                        cleaned_numbers.append(int(cleaned))
                    except:
                        continue

                matched = [val for val in required_range if val in cleaned_numbers]
                if required_range:
                    match_ratio = len(matched) / len(required_range)
                    if match_ratio >= 1.0:
                        range_match = True
                    elif match_ratio >= 0.6:
                        range_partial_match = True

            fuzzy_score = fuzz.token_set_ratio(expected_value_clean, full_pdf_text)

            # Spelling variant mismatch detection
            spelling_mismatch = None
            for am, br in spelling_variants.items():
                if am in expected_value_clean and br in full_pdf_text:
                    spelling_mismatch = f"Spelling variant mismatch ({am} vs {br})"
                    break
                if br in expected_value_clean and am in full_pdf_text:
                    spelling_mismatch = f"Spelling variant mismatch ({br} vs {am})"
                    break

            sub_values = re.findall(r'\d+\s*psi', expected_value_clean)
            sub_matches = []
            for sub in sub_values:
                sub_found = False
                for i, page_text in page_sources:
                    if sub in page_text.lower():
                        sub_found = True
                        break
                sub_matches.append(sub_found)
            all_sub_matched = len(sub_values) > 1 and all(sub_matches)

            synonym_map = {
                "electrically actuated": ["electric actuator", "electrical actuator"],
                "t-port": ["t port", "tport"],
                "3-way": ["three-way", "3 way", "3way"],
                "ball valve": ["valve", "brass valve", "ball-type valve"],
                "threaded": ["fnpt", "npt", "threaded port", "threaded connection"]
            }
            normalized_expected_parts = [p.strip() for p in re.split(r"[()\-]", expected_value_clean) if len(p.strip()) > 2]
            semantic_hits = 0
            semantic_parts_total = 0
            for phrase in normalized_expected_parts:
                phrase_variants = [phrase]
                for key, val_list in synonym_map.items():
                    if key in phrase:
                        phrase_variants.extend(val_list)
                found = False
                for variant in phrase_variants:
                    for _, page_text in page_sources:
                        if variant in page_text.lower():
                            found = True
                            break
                    if found:
                        break
                if found:
                    semantic_hits += 1
                semantic_parts_total += 1
            semantic_match = semantic_parts_total > 0 and (semantic_hits / semantic_parts_total) >= 0.75

            connection_terms = re.findall(r'(fnpt|npt|threaded|flanged|compression)', expected_value_clean)
            connection_matches = 0
            for term in connection_terms:
                if term in full_pdf_text:
                    connection_matches += 1
            connection_match = len(connection_terms) > 0 and connection_matches >= 1

            match_pdf = exact_match or normalized_match or all_sub_matched or semantic_match or connection_match or range_match or range_partial_match or (True and fuzzy_score > 80)
            if spelling_mismatch:
                match_pdf = False

            context_snippet = ""
            if exact_match:
                start_idx = full_pdf_text.find(expected_value_clean)
                context_snippet = full_pdf_text[max(0, start_idx-30):start_idx+len(expected_value_clean)+30]
            elif normalized_match:
                match_words = synonym_replace(expected_value_clean).split()
                for i, text in page_sources:
                    lower_text = synonym_replace(text.lower())
                    for sentence in lower_text.split('.'):
                        if all(word in sentence for word in match_words):
                            context_snippet = sentence.strip()
                            break
                    if context_snippet:
                        break
                if not context_snippet:
                    context_snippet = "Matched, but snippet not found"
            elif all_sub_matched:
                matched_lines = []
                for sub in sub_values:
                    for i, page_text in page_sources:
                        for line in page_text.lower().split('\n'):
                            if sub in line:
                                matched_lines.append(line.strip())
                                break
                context_snippet = " | ".join(matched_lines)
            elif semantic_match:
                context_snippet = f"Semantic components found: {semantic_hits} of {semantic_parts_total}"
            elif connection_match:
                context_snippet = f"Connection terms matched: {connection_matches}"
            elif fuzzy_score > 80:
                match_words = expected_value_clean.split()
                for i, text in page_sources:
                    for sentence in text.lower().split('.'):
                        if all(word in sentence for word in match_words):
                            context_snippet = sentence.strip()
                            break
                    if context_snippet:
                        break
            
            start_psi = end_psi = None
            matched = []
            required_range = []
            match_cases = [
                (spelling_mismatch, spelling_mismatch, f"Correct spelling to match spec sheet for: {label}"),
                (exact_match, "Exact match", "No action required"),
                (normalized_match, "Normalized match", "No action required"),
                (all_sub_matched, "Component-wise match", "No action required"),
                (semantic_match, "Semantic match", "No action required"),
                (connection_match, "Connection-type match", "No action required"),
                (range_match, f"Range values match ({start_psi}–{end_psi}) from OCR scan", "No action required"),
                (range_partial_match, f"Partial range match ({len(matched)} of {len(required_range)} values found via OCR)", "Visually review PDF for numeric consistency"),
                (fuzzy_score > 80, f"Fuzzy match ({fuzzy_score}%)", "No action required")
            ]

            for cond, note, action in match_cases:
                if cond:
                    notes = note
                    action_required = action
                    break
            else:
                notes = "Mismatch or missing"
                action_required = generate_action_required_semantic(expected_value, full_pdf_text)


            results.append({
                "Field": label, "Value": expected_value, "PDF Match": match_pdf,
                "Remarks": notes, "Match Snippet": context_snippet, "Action Required": action_required
            })

            if match_pdf:
                total_matches += 1

        # Create DataFrame
        result_df = pd.DataFrame(results)
        styled_df = result_df.style.apply(
            lambda row: ['background-color: rgba(255,0,0,0.1)' if row['PDF Match'] == False and row['Remarks'] == 'Mismatch or missing' else '' for _ in row],
            axis=1
        )

        st.markdown(f"**Summary for this product:** {total_matches} matched / {len(results)} total fields")
        st.markdown('<style>.element-container:has(.stDataFrame){height:auto!important;overflow:visible!important}.stDataFrame [data-testid="stVerticalBlock"]{max-height:none!important}</style>', unsafe_allow_html=True)

        total_height = 40 + 35 * len(result_df)

        st.dataframe(styled_df, use_container_width=True, height=total_height)

        # AI Image Verification
        if image_url:
            try:
                website_img = Image.open(BytesIO(requests.get(image_url).content)).convert("RGB")
                website_tensor = preprocess(website_img).unsqueeze(0)

                best_pdf_image = None
                best_score = -1

                # Try extracting and comparing all PDF images
                for page in doc:
                    for img in page.get_images(full=True):
                        try:
                            xref = img[0]
                            base_image = doc.extract_image(xref)
                            img_bytes = base_image["image"]
                            pil_img = Image.open(BytesIO(img_bytes)).convert("RGB")
                            pdf_tensor = preprocess(pil_img).unsqueeze(0)

                            with torch.no_grad():
                                feat1 = model(website_tensor)
                                feat2 = model(pdf_tensor)
                                score = F.cosine_similarity(feat1, feat2).item()

                            if score > best_score:
                                best_score = score
                                best_pdf_image = pil_img
                        except:
                            continue

                col_a, col_b = st.columns(2)
                col_a.image(website_img, caption="Website Image", width=250)

                if best_pdf_image:
                    col_b.image(best_pdf_image, caption="Specification Sheet Image", width=250)
                    similarity_percent = round(best_score * 100, 2)
                    if similarity_percent > 85:
                        st.success(f"Image match confidence: {similarity_percent}% ✅")
                    else:
                        st.warning(f"Image mismatch risk: only {similarity_percent}% similarity!")
                else:
                    col_b.info("No suitable image found in spec sheet for comparison.")

            except Exception as img_e:
                st.warning(f"Image verification error: {img_e}")

        # Extract website description text
        try:
            description_div = soup.select_one('div#tab-description')
            website_description_text = description_div.get_text(separator=" ").lower().strip() if description_div else ""
        except:
            website_description_text = ""

        st.markdown("### 🔎 Keyword Matches")

        keyword_variants = {
            "Pressure": ["pressure", "over pressure", "press.", "psig", "psi"],
            "Temperature": ["temperature", "temp.", "temp", "temp range", "temp range (operating)", "operating temperature"],
            "Flow": ["flow", "flowrate"],
            "Media": ["media", "fluid", "gas", "air"],
            "Weight": ["weight", "wt.", "mass"],
            "Filter": ["filter", "filtration"],
            "Material": ["material", "mat.", "composition"]
        }

        kw_results = keyword_highlights_smart(full_pdf_text, keyword_variants)
        df_kw = pd.DataFrame(kw_results)

        # Wrap and display the keyword table
        df_kw['Keyword'] = df_kw['Keyword'].apply(lambda x: '\n'.join(wrap(str(x), 25)))
        df_kw['Extracted Data'] = df_kw['Extracted Data'].apply(lambda x: '\n'.join(wrap(str(x), 50)))

        st.dataframe(df_kw, use_container_width=True, height=(40 + 35 * len(df_kw)))

        render_benefits_and_features_section(full_pdf_text)

        st.markdown("### 🚨 Warnings and Notes")
        spec_warnings = extract_critical_info_warnings_only(full_pdf_text)
        st.info(spec_warnings)

        # Save results
        csv_path = os.path.join(tempfile.gettempdir(), f"validation_report_{safe_filename}.csv")
        pdf_path_out = os.path.join(tempfile.gettempdir(), f"validation_report_{safe_filename}.pdf")

        result_df.to_csv(csv_path, index=False)
        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        ax.axis('off')
        col_labels = list(result_df.columns)
        data = result_df.copy()
        data['Field'] = data['Field'].apply(lambda x: '\n'.join(wrap(str(x), 25)))
        data['Value'] = data['Value'].apply(lambda x: '\n'.join(wrap(str(x), 30)))
        data['Remarks'] = data['Remarks'].apply(lambda x: '\n'.join(wrap(str(x), 20)))
        data['Match Snippet'] = data['Match Snippet'].apply(lambda x: '\n'.join(wrap(str(x), 40)))
        data['Action Required'] = data['Action Required'].apply(lambda x: '\n'.join(wrap(str(x), 40)))

        table = ax.table(cellText=data.values, colLabels=col_labels, cellLoc='left', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(6)
        table.scale(1, 1.6)
        for key, cell in table.get_celld().items():
            if key[0] == 0:
                cell.set_text_props(weight='bold')
            if key[1] == 0 and key[0] != 0:
                cell.set_text_props(weight='bold')
        pdf = matplotlib.backends.backend_pdf.PdfPages(pdf_path_out)
        pdf.savefig(fig, bbox_inches='tight')
        pdf.close()

        # Package all into a zip
        zip_path = os.path.join(tempfile.gettempdir(), f"valve_report_bundle_{safe_filename}.zip")
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            zipf.write(csv_path, arcname=os.path.basename(csv_path))
            zipf.write(pdf_path_out, arcname=os.path.basename(pdf_path_out))
            zipf.write(pdf_path, arcname=os.path.basename(pdf_path))

        with open(zip_path, "rb") as f:
            st.download_button(f"📥 Download Results for {readable_name}", data=f, file_name=f"valve_report_bundle_{readable_name}.zip")
            st.markdown("<hr style='border: 2px solid #bbb;'>", unsafe_allow_html=True)

except Exception as e:
    st.error(f"Error occurred: {e}")
