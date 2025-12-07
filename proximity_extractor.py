import re
from typing import Dict,Any,List,Tuple,Optional
from schema import Form83Schema
import json
from bs4 import BeautifulSoup
import html as html_module


class ProximityExtractor:
    # Proximity-based information extraction using classical IR techniques:
    # - Positional indexing for term locations
    # - Proximity matching for field extraction
    # - Section detection and boundary analysis
    # - Structural document analysis
    #
    # This implements classical IR concepts:
    # 1. Positional Indexing - track where each term appears
    # 2. Proximity Search - find values near labels
    # 3. Section Detection - identify document structure
    # 4. Pattern Matching - extract structured data

    def __init__(self):
        self.text=""
        self.lines=[]
        self.sections={}
        self.term_positions={}

        self.section_patterns=[
            (r'1\.\s*KEY INFORMATION','KEY_INFORMATION'),
            (r'2\.\s*POSITIONS','POSITIONS'),
            (r'3\.\s*DEALINGS','DEALINGS'),
            (r'4\.\s*OTHER INFORMATION','OTHER_INFORMATION'),
        ]

        # Field extraction patterns with proximity windows
        self.field_patterns = {
            'discloser_name': [
                r'\(a\)\s*Full name of discloser:?\s*(.+)',
                r'discloser:?\s*(.+)',
            ],
            'owner_controller_name': [
                r'\(b\)\s*Owner or controller.*:?\s*(.+)',
            ],
            'offeror_offeree_name': [
                r'\(c\)\s*Name of offeror/offeree.*:?\s*(.+)',
                r'Use a separate form.*:?\s*(.+)',
            ],
            'transaction_date': [
                r'\(e\)\s*Date position held.*:?\s*(\d{2}/\d{2}/\d{4})',
                r'Date.*undertaken:?\s*(\d{2}/\d{2}/\d{4})',
            ],
            'disclosure_date': [
                r'Date of disclosure:?\s*(\d{2}/\d{2}/\d{4})',
            ],
            'contact_name': [
                r'Contact name:?\s*(.+)',
            ],
            'contact_number': [
                r'Telephone number:?\s*(.+)',
            ],
        }

    def _clean_html(self, text: str) -> str:
        text = html_module.unescape(text)
        soup = BeautifulSoup(text, 'html.parser')
        text = soup.get_text()
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n', text)
        return text.strip()
    def _clean_extracted_value(self, value: str) -> str:
        if not value:
            return ""
        value = re.sub(r'<[^>]+>', '', value)
        value = html_module.unescape(value)
        value = value.replace('\\n', ' ').replace('\\t', ' ')
        value = re.sub(r'\s+', ' ', value).strip()
        value = value.strip('"\'')
        return value
    def build_positional_index(self,text: str):
        text=self._clean_html(text)
        self.text=text
        self.lines=text.split('\n')
        words=re.findall(r'\w+',text.lower())
        pos=0
        for word in words:
             if word not in self.term_positions:
                 self.term_positions[word]=[]
             self.term_positions[word].append(pos)
             pos+=len(word)+1
    def detect_sections(self)->Dict[str,Tuple[int,int]]:

        sections={}

        for i,line in enumerate(self.lines):
              for pattern,section_name in self.section_patterns:
                 if re.search(pattern,line,re.IGNORECASE):
                    sections[section_name]=i

        section_boundaries={}
        section_names=sorted(sections.keys(),key=lambda k: sections[k])

        for i,name in enumerate(section_names):
            start=sections[name]
            end=sections[section_names[i+1]] if i+1<len(section_names) else len(self.lines)
            section_boundaries[name]=(start,end)

        self.sections=section_boundaries
        return section_boundaries

    def extract_near_label(self, label_pattern: str,
                          section: Optional[str] = None,
                          max_lines: int = 3) -> str:
        lines_to_search = self.lines

        if section and section in self.sections:
            start, end = self.sections[section]
            lines_to_search = self.lines[start:end]

        for i, line in enumerate(lines_to_search):
            match = re.search(label_pattern, line, re.IGNORECASE)
            if match:
                if match.groups():
                    value = match.group(1).strip()
                    value = self._clean_extracted_value(value)
                    if value and value not in ['', 'N/A', 'None', 'n/a', 'none']:
                        return value
                for j in range(1, max_lines + 1):
                    if i + j < len(lines_to_search):
                        next_line = lines_to_search[i + j].strip()
                        next_line = self._clean_extracted_value(next_line)
                        if next_line and not next_line.startswith('('):
                            return next_line

        return ""

    def extract_table_data(self, section: str,
                          columns: List[str]) -> List[Dict[str, str]]:
        if section not in self.sections:
            return []

        start, end = self.sections[section]
        section_lines = self.lines[start:end]

        rows = []
        in_table = False

        for line in section_lines:
            if any(col.lower() in line.lower() for col in columns):
                in_table = True
                continue
            if in_table and line.strip():
                parts = re.split(r'\s{2,}|\t', line.strip())
                if len(parts) >= len(columns):
                    row = {columns[i]: parts[i].strip()
                           for i in range(min(len(columns), len(parts)))}
                    rows.append(row)
                if line.strip().startswith(('(', '1.', '2.', '3.', '4.')):
                    break

        return rows

    def extract_positions_table(self) -> Dict[str, Any]:
        if 'POSITIONS' not in self.sections:
            return {}

        start, end = self.sections['POSITIONS']
        section_text = '\n'.join(self.lines[start:end])
        security_match = re.search(r'Class of relevant security:?\s*(.+)', section_text, re.IGNORECASE)
        security_class = security_match.group(1).strip() if security_match else ""
        positions = {
            'security_class': security_class,
            'long_positions': {'number': '', 'percentage': ''},
            'short_positions': {'number': '', 'percentage': ''},}
        owned_match = re.search(r'owned.*controlled:?\s*([\d,]+)\s*([\d.]+)', section_text, re.IGNORECASE)
        if owned_match:
            positions['long_positions']['number'] = owned_match.group(1)
            positions['long_positions']['percentage'] = owned_match.group(2)
        return positions
    def extract_dealings_table(self) -> List[Dict[str, str]]:
        if 'DEALINGS' not in self.sections:
            return []
        start, end = self.sections['DEALINGS']
        section_lines = self.lines[start:end]
        dealings = []
        current_deal = {}
        for line in section_lines:
            if re.search(r'Class of relevant security:?\s*(.+)', line, re.IGNORECASE):
                match = re.search(r'Class of relevant security:?\s*(.+)', line, re.IGNORECASE)
                current_deal['security_class'] = match.group(1).strip()
            if re.search(r'Purchase/sale:?\s*(.+)', line, re.IGNORECASE):
                match = re.search(r'Purchase/sale:?\s*(.+)', line, re.IGNORECASE)
                current_deal['transaction_type'] = match.group(1).strip()
            if re.search(r'Number of securities:?\s*([\d,]+)', line, re.IGNORECASE):
                match = re.search(r'Number of securities:?\s*([\d,]+)', line, re.IGNORECASE)
                current_deal['number_of_securities'] = match.group(1).strip()
            if re.search(r'Price per unit:?\s*(.+)', line, re.IGNORECASE):
                match = re.search(r'Price per unit:?\s*(.+)', line, re.IGNORECASE)
                current_deal['price_per_unit'] = match.group(1).strip()
                if current_deal:
                    dealings.append(current_deal.copy())
                    current_deal = {}
        return dealings
    def extract(self, raw_text: str) -> Dict[str, Any]:
        self.build_positional_index(raw_text)
        self.detect_sections()
        key_info = {}
        for field, patterns in self.field_patterns.items():
            for pattern in patterns:
                value = self.extract_near_label(pattern, section='KEY_INFORMATION')
                if value:
                    key_info[field] = value
                    break
            if field not in key_info:
                key_info[field] = ""
        if 'other_parties_disclosed' not in key_info:
            other_match = re.search(r'making disclosures.*other party.*:?\s*(Yes|No)', raw_text, re.IGNORECASE)
            key_info['other_parties_disclosed'] = other_match.group(1) if other_match else ""
        if 'exempt_fund_manager_connected' not in key_info:
            key_info['exempt_fund_manager_connected'] = ""
        positions_data = self.extract_positions_table()
        dealings_data = self.extract_dealings_table()
        other_info = {
            'disclosure_date': key_info.get('disclosure_date', ''),
            'contact_name': key_info.get('contact_name', ''),
            'contact_number': key_info.get('contact_number', ''),
            'indemnity_dealing_arrangements': self.extract_near_label(
                r'Indemnity.*arrangements:?\s*(.+)', 'OTHER_INFORMATION'
            ) or "",
            'options_derivatives_agreements': self.extract_near_label(
                r'options or derivatives:?\s*(.+)', 'OTHER_INFORMATION'
            ) or "",
            'supplemental_form_attached': self.extract_near_label(
                r'Supplemental Form.*attached:?\s*(.+)', 'OTHER_INFORMATION'
            ) or "",
        }
        result = {
            'key_information': {
                'discloser_name': key_info.get('discloser_name', ''),
                'owner_controller_name': key_info.get('owner_controller_name', ''),
                'offeror_offeree_name': key_info.get('offeror_offeree_name', ''),
                'exempt_fund_manager_connected': key_info.get('exempt_fund_manager_connected', ''),
                'transaction_date': key_info.get('transaction_date', ''),
                'other_parties_disclosed': key_info.get('other_parties_disclosed', ''),
            },
            'positions': {
                'interests': {
                    'security_class': positions_data.get('security_class', ''),
                    'equity_owned_controlled': '',
                    'cash_settled_derivatives': '',
                    'stock_settled_derivatives': '',
                    'total': '',
                    'long_positions': positions_data.get('long_positions', {'number': '', 'percentage': ''}),
                    'short_positions': positions_data.get('short_positions', {'number': '', 'percentage': ''}),},
                'subscription_rights': {
                    'subscription_security_class': '',
                    'subscription_details': '',}},
            'dealings': {
                'purchases_sales': dealings_data,
                'cash_settled_derivatives': [],
                'stock_settled_derivatives_writing': [],
                'stock_settled_derivatives_exercise': [],
                'other_dealings': [],},
            'other_information': other_info,}
        return result
    def extract_sync(self, raw_text: str) -> Dict[str, Any]:
        return self.extract(raw_text)

if __name__ == "__main__":
    sample = """
    FORM 8.3

    1. KEY INFORMATION

    (a) Full name of discloser: Morgan Stanley & Co. International plc
    (c) Name of offeror/offeree in relation to whose relevant securities this form relates:
    Use a separate form for each offeror/offeree: Hipgnosis Songs Fund Limited
    (e) Date position held/dealing undertaken: 06/12/2024

    2. POSITIONS OF THE PERSON MAKING THE DISCLOSURE

    Class of relevant security: Ordinary Shares of £0.01 each

    (1) Relevant securities owned and/or controlled: 1,234,567 2.45

    3. DEALINGS (IF ANY) BY THE PERSON MAKING THE DISCLOSURE

    Class of relevant security: Ordinary Shares
    Purchase/sale: Purchase
    Number of securities: 150,000
    Price per unit: 1.1234 GBP

    4. OTHER INFORMATION

    Date of disclosure: 06/12/2024
    Contact name: John Smith
    Telephone number: +44 20 1234 5678
    """

    extractor = ProximityExtractor()
    result = extractor.extract(sample)
