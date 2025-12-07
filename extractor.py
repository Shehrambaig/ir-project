import os
import json
from typing import Dict,Any,List
import google.generativeai as genai
from schema import Form83Schema
import re


class LLMExtractor:
    def __init__(self,api_key: str=None,model: str="gemini-2.5-flash"):
        self.api_key=api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("Gemini API key required. Set GEMINI_API_KEY env variable.")

        genai.configure(api_key=self.api_key)
        self.model=genai.GenerativeModel(model)
        self.model_name=model
        
    def _preprocess_text(self,raw_text: str)->str:

        text=re.sub(r'\n{3,}','\n\n',raw_text)
        text=re.sub(r' {2,}',' ',text)
        text=text.strip()
        return text
    
    def _create_extraction_prompt(self, text: str) -> str:

        prompt = f"""You are an expert information extraction system specializing in financial regulatory forms.

**TASK**: Extract structured data from the following Form 8.3 (Rule 8 disclosure form) and convert it to JSON format.

**CRITICAL INSTRUCTIONS**:
1. Extract ALL fields according to the schema provided below
2. For missing or not-applicable fields, use empty string ""
3. Preserve exact values from the document (numbers, dates, names)
4. Return ONLY valid JSON, no additional text or explanation
5. Use nested structure as specified in the schema

**SCHEMA STRUCTURE**:

{{
  "key_information": {{
    "discloser_name": "",  // Full name of discloser
    "owner_controller_name": "",  // Owner or controller of interests
    "offeror_offeree_name": "",  // Name of offeror/offeree
    "exempt_fund_manager_connected": "",  // Exempt fund manager details
    "transaction_date": "",  // Date dealing undertaken
    "other_parties_disclosed": ""  // Yes/No
  }},
  "positions": {{
    "interests": {{
      "security_class": "",
      "equity_owned_controlled": "",
      "cash_settled_derivatives": "",
      "stock_settled_derivatives": "",
      "total": "",
      "long_positions": {{
        "number": "",
        "percentage": ""
      }},
      "short_positions": {{
        "number": "",
        "percentage": ""
      }}
    }},
    "subscription_rights": {{
      "subscription_security_class": "",
      "subscription_details": ""
    }}
  }},
  "dealings": {{
    "purchases_sales": [
      {{
        "security_class": "",
        "transaction_type": "",  // Purchase or Sale
        "number_of_securities": "",
        "price_per_unit": ""
      }}
    ],
    "cash_settled_derivatives": [
      {{
        "security_class": "",
        "product_description": "",
        "transaction_nature": "",
        "reference_securities_number": "",
        "price_per_unit": ""
      }}
    ],
    "stock_settled_derivatives_writing": [
      {{
        "security_class": "",
        "product_description": "",
        "transaction_type": "",
        "number_of_securities": "",
        "exercise_price_per_unit": "",
        "option_type": "",
        "expiry_date": "",
        "option_price_per_unit": ""
      }}
    ],
    "stock_settled_derivatives_exercise": [
      {{
        "security_class": "",
        "product_description": "",
        "transaction_type": "",
        "number_of_securities": "",
        "exercise_price_per_unit": ""
      }}
    ],
    "other_dealings": [
      {{
        "security_class": "",
        "transaction_nature": "",
        "transaction_details": "",
        "price_per_unit": ""
      }}
    ]
  }},
  "other_information": {{
    "indemnity_dealing_arrangements": "",
    "options_derivatives_agreements": "",
    "supplemental_form_attached": "",
    "disclosure_date": "",
    "contact_name": "",
    "contact_number": ""
  }}
}}

**EXTRACTION GUIDELINES**:
- Map table headers to corresponding schema fields
- Extract all rows from tables (e.g., multiple purchases/sales as array items)
- Preserve numerical precision and date formats
- If a section is marked "None" or "N/A", use empty string ""
- For arrays (purchases_sales, derivatives, etc.), include all transactions found

**DOCUMENT TO EXTRACT**:

{text}

**OUTPUT**: Return ONLY the JSON object, nothing else.
"""
        return prompt
    
    def _validate_and_clean_json(self,json_str: str)->Dict[str,Any]:
        json_str=re.sub(r'^```json\s*','',json_str)
        json_str=re.sub(r'\s*```$','',json_str)
        json_str=json_str.strip()
        parsed=json.loads(json_str)
        return parsed

    def _apply_schema_validation(self,data: Dict[str,Any])->Form83Schema:
        validated=Form83Schema(**data)
        return validated

    async def extract(self,raw_text: str,temperature: float=0.0)->Dict[str,Any]:
        preprocessed=self._preprocess_text(raw_text)
        prompt=self._create_extraction_prompt(preprocessed)

        generation_config={
            "temperature":temperature,
            "max_output_tokens":8192,
        }

        full_prompt="You are an expert data extraction system. Return only valid JSON.\n\n"+prompt

        response=self.model.generate_content(
            full_prompt,
            generation_config=generation_config)

        llm_output=response.text
        parsed_data=self._validate_and_clean_json(llm_output)
        validated_schema=self._apply_schema_validation(parsed_data)

        return validated_schema.model_dump()
    
    def extract_sync(self, raw_text: str, temperature: float = 0.0) -> Dict[str, Any]:
        import asyncio
        return asyncio.run(self.extract(raw_text, temperature))


class PostProcessor:

    @staticmethod
    def validate_date_format(date_str: str) -> bool:
        patterns = [
            r'^\d{2}/\d{2}/\d{4}$',
            r'^\d{4}-\d{2}-\d{2}$',
        ]
        return any(re.match(p, date_str) for p in patterns) if date_str else True
    
    @staticmethod
    def validate_percentage(pct_str: str) -> bool:
        if not pct_str:
            return True
        return bool(re.match(r'^\d+\.?\d*%?$', pct_str.replace(',', '')))
    
    @staticmethod
    def clean_numeric(num_str: str) -> str:
        if not num_str:
            return ""
        return num_str.replace(',', '')
