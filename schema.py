from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import date


# Section 1: Key Info
class KeyInformation(BaseModel):
    discloser_name: str = Field(default="", description="Full name of discloser")
    owner_controller_name: str = Field(default="", description="Owner or controller of interests")
    offeror_offeree_name: str = Field(default="", description="Name of offeror/offeree")
    exempt_fund_manager_connected: str = Field(default="", description="Exempt fund manager details")
    transaction_date: str = Field(default="", description="Date dealing undertaken")
    other_parties_disclosed: str = Field(default="", description="Other parties disclosed (Yes/No)")


# Section 2:Positions
class InterestsLongPosition(BaseModel):
    number: str = Field(default="", description="Number of securities")
    percentage: str = Field(default="", description="Percentage of securities")


class InterestsShortPosition(BaseModel):
    number: str = Field(default="", description="Number of securities")
    percentage: str = Field(default="", description="Percentage of securities")


class InterestsPosition(BaseModel):
    security_class: str = Field(default="", description="Class of relevant security")
    equity_owned_controlled: str = Field(default="", description="Equity owned/controlled")
    cash_settled_derivatives: str = Field(default="", description="Cash-settled derivatives")
    stock_settled_derivatives: str = Field(default="", description="Stock-settled derivatives")
    total: str = Field(default="", description="Total interests")
    long_positions: InterestsLongPosition = Field(default_factory=InterestsLongPosition)
    short_positions: InterestsShortPosition = Field(default_factory=InterestsShortPosition)


class SubscriptionRights(BaseModel):
    subscription_security_class: str = Field(default="", description="Security class for subscription")
    subscription_details: str = Field(default="", description="Details and percentages")


class Positions(BaseModel):
    interests: InterestsPosition = Field(default_factory=InterestsPosition)
    subscription_rights: SubscriptionRights = Field(default_factory=SubscriptionRights)


# Section 3: Dealings
class PurchaseSale(BaseModel):
    security_class: str = Field(default="", description="Class of security")
    transaction_type: str = Field(default="", description="Purchase or Sale")
    number_of_securities: str = Field(default="", description="Number of securities")
    price_per_unit: str = Field(default="", description="Price per unit")


class CashSettledDerivative(BaseModel):
    security_class: str = Field(default="", description="Security class")
    product_description: str = Field(default="", description="Product description")
    transaction_nature: str = Field(default="", description="Nature of dealing")
    reference_securities_number: str = Field(default="", description="Number of reference securities")
    price_per_unit: str = Field(default="", description="Price per unit")


class StockSettledDerivativeWriting(BaseModel):
    security_class: str = Field(default="", description="Security class")
    product_description: str = Field(default="", description="Product description")
    transaction_type: str = Field(default="", description="Writing/purchasing/selling/varying")
    number_of_securities: str = Field(default="", description="Number of securities")
    exercise_price_per_unit: str = Field(default="", description="Exercise price")
    option_type: str = Field(default="", description="Call/Put option type")
    expiry_date: str = Field(default="", description="Expiry date")
    option_price_per_unit: str = Field(default="", description="Option money paid/received")


class StockSettledDerivativeExercise(BaseModel):
    security_class: str = Field(default="", description="Security class")
    product_description: str = Field(default="", description="Product description (e.g., call option)")
    transaction_type: str = Field(default="", description="Exercising/exercised against")
    number_of_securities: str = Field(default="", description="Number of securities")
    exercise_price_per_unit: str = Field(default="", description="Exercise price")


class OtherDealings(BaseModel):
    security_class: str = Field(default="", description="Security class")
    transaction_nature: str = Field(default="", description="Nature of dealing (e.g., subscription)")
    transaction_details: str = Field(default="", description="Transaction details")
    price_per_unit: str = Field(default="", description="Price per unit if applicable")


class Dealings(BaseModel):
    purchases_sales: List[PurchaseSale] = Field(default_factory=list)
    cash_settled_derivatives: List[CashSettledDerivative] = Field(default_factory=list)
    stock_settled_derivatives_writing: List[StockSettledDerivativeWriting] = Field(default_factory=list)
    stock_settled_derivatives_exercise: List[StockSettledDerivativeExercise] = Field(default_factory=list)
    other_dealings: List[OtherDealings] = Field(default_factory=list)


# Section 4:  Other Infomation
class OtherInformation(BaseModel):
    indemnity_dealing_arrangements: str = Field(default="", description="Indemnity and dealing arrangements")
    options_derivatives_agreements: str = Field(default="", description="Options/derivatives agreements")
    supplemental_form_attached: str = Field(default="", description="Is supplemental form attached")
    disclosure_date: str = Field(default="", description="Date of disclosure")
    contact_name: str = Field(default="", description="Contact name")
    contact_number: str = Field(default="", description="Contact telephone number")


# Complete Form 8.3
class Form83Schema(BaseModel):
    key_information: KeyInformation = Field(default_factory=KeyInformation)
    positions: Positions = Field(default_factory=Positions)
    dealings: Dealings = Field(default_factory=Dealings)
    other_information: OtherInformation = Field(default_factory=OtherInformation)
