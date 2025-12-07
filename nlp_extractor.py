import re
from typing import Dict,Any,List,Tuple,Optional
import spacy
from spacy.matcher import Matcher
import nltk
from nltk.tokenize import sent_tokenize,word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from bs4 import BeautifulSoup
import html as html_module
from collections import defaultdict


class NLPExtractor:

    def __init__(self):
        self.nlp=spacy.load("en_core_web_sm")
        nltk.data.find('tokenizers/punkt')

        self.doc=None
        self.text=""
        self.sentences=[]
        self.entities = []
        self.tfidf_scores = {}

    def _clean_text(self, text: str) -> str:
        text = html_module.unescape(text)
        soup = BeautifulSoup(text, 'html.parser')
        text = soup.get_text()
        text = text.replace('\\n', '\n').replace('\\t', ' ')
        text = re.sub(r'([Tt]elephone number)\s*[\*†#]+\s*:', r'\1:', text)
        text = re.sub(r'([Cc]ontact name)\s*[\*†#]+\s*:', r'\1:', text)
        text = re.sub(r'(\d+)\.\s+([A-Z\s]+)', r'\1. \2', text)
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        lines = text.split('\n')
        normalized_lines = []
        for line in lines:
            line = re.sub(r'[ \t]+', ' ', line)
            normalized_lines.append(line.strip())
        text = '\n'.join(normalized_lines)
        return text.strip()
    def _is_noise_line(self, line: str) -> bool:
        if not line or len(line.strip()) < 3:
            return True
        line_lower = line.lower().strip()
        noise_patterns = [
            r'^the naming of nominee',
            r'^for a trust',
            r'^use a separate form',
            r'^if it is a cash offer',
            r'^if there are',
            r'^details of any',
            r'^all interests and all short',
            r'^where there have been',
            r'^the currency of all prices',
            r'^irrevocable commitments',
            r'^if there are no such',
            r'should not be included',
            r'^public disclosures under',
            r'^\*if the discloser',
            r'^this information is provided',
            r'^rns may use',
            r'^end$',
        ]

        for pattern in noise_patterns:
            if re.search(pattern, line_lower):
                return True
        if re.match(r'^-{3,}$', line) or re.match(r'^={3,}$', line):
            return True
        return False
    def _is_na_value(self, value: str) -> bool:
        if not value:
            return True
        value_clean = value.strip().upper()
        na_values = ['N/A', 'NA', 'NONE', 'NIL', 'NOT APPLICABLE', '-']
        return value_clean in na_values
    def _parse_positions_table(self, section_text: str) -> Dict[str, Any]:
        result = {
            'security_class': '',
            'long_positions': {'number': '', 'percentage': ''},
            'short_positions': {'number': '', 'percentage': ''},
            'owned_controlled': {'number': '', 'percentage': ''},
            'cash_derivatives': {'number': '', 'percentage': ''},
            'stock_derivatives': {'number': '', 'percentage': ''},
        }

        lines = section_text.split('\n')

        for i, line in enumerate(lines):
            if re.search(r'Class of relevant security', line, re.IGNORECASE):
                for j in range(i+1, min(i+5, len(lines))):
                    next_line = lines[j].strip()
                    if next_line and not re.match(r'^\s*$', next_line):
                        if 'interests' not in next_line.lower() and 'number' not in next_line.lower():
                            result['security_class'] = next_line
                            break
                break
        for i, line in enumerate(lines):
            line_lower = line.lower()

            # Helper function to collect numbers from next N lines
            def collect_numbers_ahead(start_idx, max_lines=5):
                """Collect all numbers from the next few lines, skipping row markers"""
                nums = []
                found_first_number = False

                for j in range(start_idx, min(start_idx + max_lines, len(lines))):
                    current_line = lines[j]
                    if j > start_idx:
                        if re.match(r'\s*\(\d\)', current_line) or re.match(r'^\d+\.', current_line):
                            break
                        starting_line = lines[start_idx].lower()
                        if not re.match(r'\s*total\s*:', starting_line) and re.match(r'\s*total\s*:', current_line.lower()):
                            break
                    clean_line = re.sub(r'\(\d+\)', '', current_line)
                    line_nums = re.findall(r'(?:[\d,]+(?:\.\d+)?|\.\d+)', clean_line)
                    if line_nums:
                        nums.extend(line_nums)
                        found_first_number = True
                        if len(nums) >= 2:
                            break
                    elif found_first_number and not line_nums and current_line.strip() == '':
                        continue
                return nums
            if re.search(r'\(1\).*(?:relevant|owned|controlled)', line_lower):
                numbers = collect_numbers_ahead(i, max_lines=8)
                if len(numbers) >= 2:
                    result['owned_controlled']['number'] = numbers[0]
                    pct = numbers[1]
                    if pct.startswith('.'):
                        pct = pct
                    elif not pct.endswith('%') and '.' in pct and float(pct) < 100:
                        pct = pct
                    result['owned_controlled']['percentage'] = pct
            elif re.search(r'\(2\).*cash', line_lower):
                numbers = collect_numbers_ahead(i, max_lines=8)
                if len(numbers) >= 2:
                    result['cash_derivatives']['number'] = numbers[0]
                    pct = numbers[1]
                    if pct.startswith('.'):
                        pct = pct
                    elif not pct.endswith('%') and '.' in pct and float(pct) < 100:
                        pct = pct
                    result['cash_derivatives']['percentage'] = pct
            elif re.search(r'\(3\).*stock', line_lower):
                numbers = collect_numbers_ahead(i, max_lines=8)
                if len(numbers) >= 2:
                    result['stock_derivatives']['number'] = numbers[0]
                    pct = numbers[1]
                    if pct.startswith('.'):
                        pct = pct
                    elif not pct.endswith('%') and '.' in pct and float(pct) < 100:
                        pct = pct
                    result['stock_derivatives']['percentage'] = pct

            elif re.search(r'total\s*:', line_lower):
                numbers = collect_numbers_ahead(i, max_lines=15)
                if len(numbers) >= 2:
                    result['long_positions']['number'] = numbers[0]
                    pct = numbers[1]
                    if pct.startswith('.'):
                        pct = pct
                    elif not pct.endswith('%') and '.' in pct and float(pct) < 100:
                        pct = pct
                    result['long_positions']['percentage'] = pct

        return result

    def _compute_tfidf(self,text: str):
        sentences=sent_tokenize(text)

        if len(sentences)<2:
            return

        vectorizer=TfidfVectorizer(max_features=100,stop_words='english')
        tfidf_matrix=vectorizer.fit_transform(sentences)
        feature_names=vectorizer.get_feature_names_out()
        avg_scores=tfidf_matrix.mean(axis=0).A1
        self.tfidf_scores=dict(zip(feature_names,avg_scores))

    def _extract_entities(self):
        self.entities = {
            'ORG': [],
             'PERSON': [],
            'DATE': [],
             'MONEY': [],
            'CARDINAL': [],
              'PERCENT': [],
        }

        for ent in self.doc.ents:
            if ent.label_ in self.entities:
                self.entities[ent.label_].append({
                    'text': ent.text,
                    'start': ent.start_char,
                    'end': ent.end_char
                })

    def _find_pattern_in_context(self, pattern: str, context_window: int = 100) -> str:
        match = re.search(pattern, self.text, re.IGNORECASE)
        if not match:
            return ""
        if match.groups():
            value = match.group(1).strip()
            value = re.sub(r'<[^>]+>', '', value)
            value = re.sub(r'\s+', ' ', value)
            return value.strip()
        start = match.end()
        end = min(start + context_window, len(self.text))
        context = self.text[start:end]
        lines = context.split('\n')
        for line in lines:
            line = line.strip()
            if line and not line.startswith('('):
                line = re.sub(r'<[^>]+>', '', line)
                return line.strip()

        return ""

    def _extract_by_label_proximity(self, labels: List[str], entity_type: Optional[str] = None, max_lines_ahead: int = 10) -> str:
        for label in labels:
            lines = self.text.split('\n')
            for i, line in enumerate(lines):
                if re.search(label, line, re.IGNORECASE):
                    pattern = label + r':?\s*(.+?)$'
                    match = re.search(pattern, line, re.IGNORECASE)
                    if match:
                        value = match.group(1).strip()
                        value = re.sub(r'<[^>]+>', '', value)
                        value = re.sub(r'\s+', ' ', value)
                        if value and len(value) > 2 and not self._is_na_value(value):
                            if not re.match(r'^\([a-z]\)', value) and not value.lower().startswith('use '):
                                return value.strip()
                    for j in range(1, min(max_lines_ahead + 1, len(lines) - i)):
                        next_line = lines[i + j].strip()
                        if re.match(r'^\(\w\)', next_line) or re.match(r'^\d+\.', next_line):
                            break
                        if self._is_noise_line(next_line):
                            continue
                        if not next_line:
                            continue
                        if self._is_na_value(next_line):
                            return ""
                        value = re.sub(r'<[^>]+>', '', next_line)
                        value = re.sub(r'\s+', ' ', value)
                        if entity_type and entity_type in self.entities:
                            # Check if any entity of this type appears in this line
                            for ent_info in self.entities[entity_type]:
                                ent_text = ent_info['text']
                                if ent_text in next_line:
                                    return ent_text
                        return value.strip()

        return ""

    def _extract_first_entity(self, entity_type: str) -> str:
        if entity_type in self.entities and self.entities[entity_type]:
            return self.entities[entity_type][0]['text']
        return ""

    def _extract_numbers_with_context(self, section_keywords: List[str] = None) -> Dict[str, Any]:
        numbers_data = {
            'share_counts': [],
            'percentages': [],
            'prices': []
        }
        share_keywords = ['securities', 'shares', 'owned', 'controlled', 'number', 'total', 'relevant']
        price_keywords = ['price', 'unit', 'gbp', 'usd', 'eur', 'pence', 'per']
        percentage_keywords = ['percent', '%', 'percentage']

        for sent in self.doc.sents:
            if section_keywords:
                if not any(kw.lower() in sent.text.lower() for kw in section_keywords):
                    continue

            sent_text_lower = sent.text.lower()

            for token in sent:
                # Check for numbers (CARDINAL entities, MONEY entities, or numeric tokens)
                if token.ent_type_ in ['CARDINAL', 'MONEY'] or token.like_num:
                    # Get context window (5 tokens before and after)
                    start_idx = max(0, token.i - 5)
                    end_idx = min(len(self.doc), token.i + 6)
                    context_tokens = [self.doc[i].text.lower() for i in range(start_idx, end_idx)]
                    context_text = ' '.join(context_tokens)

                    number_value = token.text.replace(',', '')
                    if any(kw in context_text for kw in share_keywords):
                        num_val = float(number_value.replace(',', ''))
                        if num_val > 100 or ',' in token.text:
                            numbers_data['share_counts'].append({
                                'value': token.text,
                                'context': sent.text[:150],
                                'position': token.i})
                    elif any(kw in context_text for kw in price_keywords):
                        numbers_data['prices'].append({
                            'value': token.text,
                            'context': sent.text[:150],
                            'position': token.i})
                if token.ent_type_ == 'PERCENT' or '%' in token.text:
                    numbers_data['percentages'].append({
                        'value': token.text,
                        'context': sent.text[:150],
                        'position': token.i})

        return numbers_data

    def _extract_using_dependencies(self, target_label: str, target_type: str = 'CARDINAL') -> List[Dict[str, str]]:
        results = []
        for sent in self.doc.sents:
            if target_label.lower() in sent.text.lower():
                for ent in sent.ents:
                    if ent.label_ == target_type:
                        results.append({
                            'value': ent.text,
                            'label': target_label,
                            'sentence': sent.text})
                for token in sent:
                    if token.like_num and target_type == 'CARDINAL':
                        distance_to_label = min(
                            abs(token.i - t.i)
                            for t in sent
                            if target_label.lower() in t.text.lower()
                        ) if any(target_label.lower() in t.text.lower() for t in sent) else float('inf')

                        if distance_to_label <= 5:  # Within 5 tokens
                            results.append({
                                'value': token.text,
                                'label': target_label,
                                'sentence': sent.text})
        return results
    def _detect_table_structure(self, section_text: str) -> Dict[str, Any]:
        lines = section_text.split('\n')
        table_structure = {
            'header_line': -1,
            'data_start': -1,
            'columns': [],
            'rows': []}
        header_patterns = [
            r'(?:Class|Security).*(?:Purchase|Sale|Type).*(?:Number|Quantity).*(?:Price)',
            r'Security.*Transaction.*Number.*Price',
            r'Interests.*Short.*Number.*%',]
        for i, line in enumerate(lines):
            for pattern in header_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    table_structure['header_line'] = i
                    table_structure['columns'] = re.split(r'\s{2,}|\t', line.strip())
                    table_structure['data_start'] = i + 1
                    break
            if table_structure['header_line'] >= 0:
                break
        if table_structure['header_line'] >= 0:
            for i in range(table_structure['data_start'], len(lines)):
                line = lines[i].strip()
                if re.match(r'^\d+\.', line) or (not line and i > table_structure['data_start'] + 2):
                    break
                if not line:
                    continue
                cells = re.split(r'\s{2,}|\t', line)
                if len(cells) >= 2:
                    table_structure['rows'].append(cells)
        return table_structure
    def _extract_table_using_structure(self, section_text: str) -> List[Dict[str, str]]:
        transactions = []
        table_fields = {
            'security_class': r'(?:Class of (?:relevant )?security)[:\s]*([^\n]+)',
            'transaction_type': r'(?:Purchase|Sale)[/\s]*:?\s*(Purchase|Sale)',
            'number_of_securities': r'Number of securities[:\s]*([\d,]+)',
            'price_per_unit': r'Price per unit[:\s]*([\d.]+\s*\w*)'}
        table_struct = self._detect_table_structure(section_text)
        if table_struct['rows']:
            for row in table_struct['rows']:
                if all(self._is_na_value(cell) for cell in row):
                    continue
                if len(row) >= 3:
                    transaction = {
                        'security_class': row[0] if len(row) > 0 and not self._is_na_value(row[0]) else '',
                        'transaction_type': row[1] if len(row) > 1 and not self._is_na_value(row[1]) else '',
                        'number_of_securities': row[2] if len(row) > 2 and not self._is_na_value(row[2]) else '',
                        'price_per_unit': row[3] if len(row) > 3 and not self._is_na_value(row[3]) else ''}
                    if any(val for val in transaction.values()):
                        transactions.append(transaction)
        else:
            transaction_blocks = re.split(
                r'(?=Class of (?:relevant )?security)',
                section_text,
                flags=re.IGNORECASE)
            for block in transaction_blocks:
                if not block.strip():
                    continue
                transaction = {}
                for field_name, pattern in table_fields.items():
                    match = re.search(pattern, block, re.IGNORECASE)
                    if match:
                        value = match.group(1).strip() if match.groups() else match.group(0)
                        # Clean extracted value
                        value = re.sub(r'\s+', ' ', value)
                        value = re.sub(r'<[^>]+>', '', value)  # Remove HTML
                        value = value.strip('"\'')
                        if not self._is_na_value(value):
                            transaction[field_name] = value
                if len(transaction) >= 2:
                    # Check that we have actual data, not just column headers
                    has_real_data = any(
                        field in transaction and
                        transaction[field] and
                        not transaction[field].lower() in ['purchase/sale', 'number of securities', 'price per unit']
                        for field in ['number_of_securities', 'price_per_unit'])
                    if has_real_data:
                        transactions.append(transaction)
        return transactions
    def _extract_table_like_data(self, section_start_pattern: str) -> List[Dict[str, str]]:
        dealings = []
        section_match = re.search(section_start_pattern, self.text, re.IGNORECASE)
        if not section_match:
            return dealings
        section_text = self.text[section_match.end():]
        patterns = [
            r'(?:Class of relevant security|Security)[:\s]+(.+?)[\n\r]',
            r'(?:Purchase|Sale)[:\s]+(\w+)',
            r'(?:Number of securities)[:\s]+([\d,]+)',
            r'(?:Price per unit)[:\s]+([\d.,]+\s*\w*)',
        ]
        lines = section_text.split('\n')
        current_transaction = {}
        for line in lines:
            line = line.strip()
            if re.match(r'^\d+\.', line) or line.startswith('4.'):
                break
            if re.search(r'(?:Class of relevant security|Security)', line, re.IGNORECASE):
                match = re.search(r'(?:Class of relevant security|Security)[:\s]+(.+)', line, re.IGNORECASE)
                if match:
                    if current_transaction:
                        dealings.append(current_transaction)
                    current_transaction = {'security_class': match.group(1).strip()}
            elif re.search(r'(?:Purchase|Sale)', line, re.IGNORECASE):
                match = re.search(r'(Purchase|Sale)', line, re.IGNORECASE)
                if match and 'security_class' in current_transaction:
                    current_transaction['transaction_type'] = match.group(1)
            elif re.search(r'Number of securities', line, re.IGNORECASE):
                match = re.search(r'Number of securities[:\s]+([\d,]+)', line, re.IGNORECASE)
                if match and 'security_class' in current_transaction:
                    current_transaction['number_of_securities'] = match.group(1).strip()
            elif re.search(r'Price per unit', line, re.IGNORECASE):
                match = re.search(r'Price per unit[:\s]+([\d.,]+\s*\w*)', line, re.IGNORECASE)
                if match and 'security_class' in current_transaction:
                    current_transaction['price_per_unit'] = match.group(1).strip()
                    if current_transaction:
                        dealings.append(current_transaction)
                        current_transaction = {}
        if current_transaction and len(current_transaction) > 1:
            dealings.append(current_transaction)
        return dealings
    def extract(self, raw_text: str) -> Dict[str, Any]:
        # Main NLP-powered extraction pipeline
        # Steps:
        # 1. Text cleaning and preprocessing
        # 2. spaCy NLP processing (tokenization, NER, POS, dependencies)
        # 3. NLTK sentence tokenization
        # 4. TF-IDF computation
        # 5. Entity extraction
        # 6. Pattern-based + NER-based field extraction
        self.text = self._clean_text(raw_text)
        self.doc = self.nlp(self.text)
        self.sentences = sent_tokenize(self.text)
        self._compute_tfidf(self.text)
        self._extract_entities()

        discloser_name = self._extract_by_label_proximity([r'\(a\)\s*Full name of discloser'])
        if not discloser_name and self.entities['ORG']:
            discloser_name = self.entities['ORG'][0]['text']

        offeror_offeree = self._extract_by_label_proximity([r'\(c\)\s*Name of offeror'])
        if not offeror_offeree or 'relation to' in offeror_offeree.lower():
            offeror_match = re.search(r'each offeror/offeree[:\s]*(.+?)[\n\(]', self.text, re.IGNORECASE)
            if offeror_match:
                offeror_offeree = offeror_match.group(1).strip()
            elif len(self.entities['ORG']) > 1:
                offeror_offeree = self.entities['ORG'][1]['text']

        transaction_date = self._extract_by_label_proximity([r'\(e\)\s*Date position held'])
        if not transaction_date or 'undertaken' in transaction_date.lower():
            date_match = re.search(r'\(e\)[^\n]+\n[^\n]*(\d{2}/\d{2}/\d{4})', self.text, re.IGNORECASE)
            if date_match:
                transaction_date = date_match.group(1)
            elif self.entities['DATE']:
                transaction_date = self.entities['DATE'][0]['text']

        other_parties = self._find_pattern_in_context(r'other party.*?[:\s]+(Yes|No)', 50)


        positions_section_match = re.search(r'2\.\s*POSITIONS.*?(?=3\.|$)', self.text, re.DOTALL | re.IGNORECASE)

        if positions_section_match:
            positions_section_text = positions_section_match.group(0)
            positions_data = self._parse_positions_table(positions_section_text)
            security_class = positions_data['security_class']
            long_number = positions_data['long_positions']['number']
            long_percentage = positions_data['long_positions']['percentage']
            equity_owned_controlled = positions_data['owned_controlled']
            cash_settled = positions_data['cash_derivatives']
            stock_settled = positions_data['stock_derivatives']
        else:
            security_class = self._extract_by_label_proximity([r'Class of relevant security'], max_lines_ahead=5)
            positions_numbers = self._extract_numbers_with_context(['POSITIONS', 'owned', 'controlled', 'relevant securities'])
            long_number = ""
            long_percentage = ""
            if positions_numbers['share_counts']:
                sorted_shares = sorted(positions_numbers['share_counts'], key=lambda x: x['position'])
                for share_data in sorted_shares:
                    if 'owned' in share_data['context'].lower() or 'controlled' in share_data['context'].lower():
                        long_number = share_data['value']
                        break
                if not long_number and sorted_shares:
                    long_number = sorted_shares[0]['value']
            if positions_numbers['percentages']:
                real_percentages = [p for p in positions_numbers['percentages'] if re.search(r'\d', p['value'])]
                if real_percentages:
                    long_percentage = real_percentages[0]['value']
            equity_owned_controlled = {'number': '', 'percentage': ''}
            cash_settled = {'number': '', 'percentage': ''}
            stock_settled = {'number': '', 'percentage': ''}
        dealings_section_match = re.search(r'3\.\s*DEALINGS.*?(?=4\.|$)', self.text, re.DOTALL | re.IGNORECASE)
        if dealings_section_match:
            dealings_section_text = dealings_section_match.group(0)
            dealings = self._extract_table_using_structure(dealings_section_text)
            if not dealings:
                dealings = self._extract_table_like_data(r'3\.\s*DEALINGS')
        else:
            dealings = self._extract_table_like_data(r'3\.\s*DEALINGS')
        disclosure_date = (
            self._extract_by_label_proximity([r'Date of disclosure'], 'DATE') or
            (self.entities['DATE'][1]['text'] if len(self.entities['DATE']) > 1 else ""))
        contact_name = self._extract_by_label_proximity([r'Contact name'], 'PERSON')
        contact_number = self._extract_by_label_proximity([r'Telephone number'])
        result = {
            'key_information': {
                'discloser_name': discloser_name,
                'owner_controller_name': '',
                'offeror_offeree_name': offeror_offeree,
                'exempt_fund_manager_connected': '',
                'transaction_date': transaction_date,
                'other_parties_disclosed': other_parties.lower() if other_parties else 'no',
            },
            'positions': {
                'interests': {
                    'security_class': security_class,
                    'equity_owned_controlled': f"{equity_owned_controlled['number']} ({equity_owned_controlled['percentage']})" if equity_owned_controlled['number'] else '',
                    'cash_settled_derivatives': f"{cash_settled['number']} ({cash_settled['percentage']})" if cash_settled['number'] else '',
                    'stock_settled_derivatives': f"{stock_settled['number']} ({stock_settled['percentage']})" if stock_settled['number'] else '',
                    'total': f"{long_number} ({long_percentage})" if long_number else '',
                    'long_positions': {
                        'number': long_number,
                        'percentage': long_percentage,
                    },
                    'short_positions': {
                        'number': '',
                        'percentage': '',
                    },
                    # Detailed breakdown (bonus info)
                    'breakdown': {
                        'owned_controlled': equity_owned_controlled,
                        'cash_derivatives': cash_settled,
                        'stock_derivatives': stock_settled,
                    }
                },
                'subscription_rights': {
                    'subscription_security_class': '',
                    'subscription_details': '',
                }
            },
            'dealings': {
                'purchases_sales': dealings,
                'cash_settled_derivatives': [],
                'stock_settled_derivatives_writing': [],
                'stock_settled_derivatives_exercise': [],
                'other_dealings': [],
            },
            'other_information': {
                'disclosure_date': disclosure_date,
                'contact_name': contact_name,
                'contact_number': contact_number,
                'indemnity_dealing_arrangements': '',
                'options_derivatives_agreements': '',
                'supplemental_form_attached': '',
            }
        }

        return result


# Test
if __name__ == "__main__":
    sample = """
    FORM 8.3

    1. KEY INFORMATION
    (a) Full name of discloser: Morgan Stanley & Co. International plc
    (c) Name of offeror/offeree: Hipgnosis Songs Fund Limited
    (e) Date position held/dealing undertaken: 06/12/2024

    2. POSITIONS
    Class of relevant security: Ordinary Shares
    Long positions: 1,234,567 shares (2.45%)

    3. DEALINGS
    Class of relevant security: Ordinary Shares
    Purchase/Sale: Purchase
    Number of securities: 150,000
    Price per unit: 1.23 GBP

    4. OTHER INFORMATION
    Date of disclosure: 06/12/2024
    Contact name: John Smith
    Telephone number: +44 20 1234 5678
    """

    extractor = NLPExtractor()
    result = extractor.extract(sample)
