import time
from typing import Dict, Any
from nlp_extractor import NLPExtractor
from extractor import LLMExtractor
from schema import Form83Schema
class HybridExtractor:


    def __init__(self, api_key: str = None, llm_model: str = "gemini-2.5-flash"):
        self.nlp_extractor = NLPExtractor()
        self.llm_extractor = LLMExtractor(api_key=api_key, model=llm_model)

    def _create_llm_focused_prompt(self, text: str) -> str:
        prompt = f"""You are an expert at extracting structured table data from financial regulatory forms.

**TASK**: Extract the "positions" and "dealings" sections from this Form 8.3 (Rule 8 disclosure form).

**CRITICAL INSTRUCTIONS**:
1. Extract ONLY positions and dealings tables - ignore other fields
2. For missing or not-applicable fields, use empty string ""
3. Preserve EXACT values from tables (numbers with commas, dates, percentages with decimals)
4. Return ONLY valid JSON, no additional text, no markdown
5. Handle multi-line table cells and nested headers correctly
6. Extract ALL rows from tables (don't skip any)

**COMPLETE SCHEMA TO FILL**:

{{
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
      }},
      "breakdown": {{
        "owned_controlled": {{
          "number": "",
          "percentage": ""
        }},
        "cash_derivatives": {{
          "number": "",
          "percentage": ""
        }},
        "stock_derivatives": {{
          "number": "",
          "percentage": ""
        }}
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
        "transaction_type": "",
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
  }}
}}

**DETAILED EXTRACTION RULES**:

1. **POSITIONS (Section 2)**:
   - Extract security class from the table header
   - Extract numbers from BOTH "Number" and "%" columns
   - Look for TOTAL row for long_positions (total number and total percentage)
   - Extract breakdown rows if present:
     * (1) Relevant securities owned and/or controlled
     * (2) Cash-settled derivatives
     * (3) Stock-settled derivatives (including options) and agreements to purchase/sell
   - Preserve comma formatting: "1,322,685" not "1322685"
   - Preserve percentage decimals: ".63" or "1.27" not "0.63" or "1"

2. **DEALINGS (Section 3)**:
   - Extract ALL transaction rows from each table
   - If table says "None" or all rows are "N/A", use empty array []
   - Common tables: Purchases/sales, Cash-settled derivatives, Stock-settled derivatives
   - Extract: security class, type, quantity, price for each transaction
   - If multiple transactions, include ALL in the array

**FULL DOCUMENT TEXT**:

{text}

**OUTPUT**: Return ONLY the JSON object with positions and dealings data. No explanations, no markdown formatting, just pure JSON.
"""
        return prompt

    async def extract(self, raw_text: str, verbose: bool = False) -> Dict[str, Any]:
        import re
        import json

        nlp_result = self.nlp_extractor.extract(raw_text)
        focused_prompt = self._create_llm_focused_prompt(raw_text)
        generation_config = {
            "temperature": 0.0,
            "max_output_tokens": 8192,}
        response = self.llm_extractor.model.generate_content(
            focused_prompt,
            generation_config=generation_config)
        llm_output = response.text
        llm_output = re.sub(r'^```json\s*', '', llm_output)
        llm_output = re.sub(r'\s*```$', '', llm_output)
        llm_output = llm_output.strip()
        llm_data = json.loads(llm_output)
        merged_result = {
            "key_information": nlp_result["key_information"],
            "other_information": nlp_result["other_information"],
            "positions": llm_data["positions"],
            "dealings": llm_data["dealings"]}
        validated = Form83Schema(**merged_result)
        return validated.model_dump()
    def extract_sync(self, raw_text: str, verbose: bool = False) -> Dict[str, Any]:
        import asyncio
        return asyncio.run(self.extract(raw_text, verbose=verbose))
# Convenience function
def extract_hybrid(raw_text: str, api_key: str = None, verbose: bool = True) -> Dict[str, Any]:
    extractor = HybridExtractor(api_key=api_key)
    return extractor.extract_sync(raw_text, verbose=verbose)
