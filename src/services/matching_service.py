"""
Marketplace Matching Service
Matches buyers, sellers, and investors based on location, capacity, financial profiles.
"""

from typing import List, Dict, Any, Optional, Tuple
import math
from datetime import datetime
from pydantic import BaseModel
from src.utils.logger import get_logger

logger = get_logger(__name__)


class UserProfile(BaseModel):
    """User profile for matching"""
    user_id: str
    user_type: str  # "seller", "buyer", "investor"
    latitude: float
    longitude: float
    capacity_kw: Optional[float] = None  # For sellers
    demand_kw: Optional[float] = None  # For buyers
    budget_rupees: Optional[float] = None  # For buyers/investors
    financial_score: float  # 0-100
    credit_rating: str  # "excellent", "good", "fair", "poor"
    available_balance: float  # Current capital
    max_investment: Optional[float] = None  # For investors
    roi_expectation: Optional[float] = None  # Expected ROI percentage
    system_age_years: Optional[float] = None  # For sellers
    system_efficiency: Optional[float] = None  # For sellers, 0-100%
    willing_to_travel_km: float = 25  # Max distance


class MatchingResult(BaseModel):
    """Match result between two users"""
    match_id: str
    user_a_id: str
    user_b_id: str
    match_score: float  # 0-100
    distance_km: float
    compatibility_reason: str
    financial_feasibility: Dict[str, Any]  # Price negotiation, payment terms
    transaction_risk: str  # "low", "medium", "high"
    recommended_price_rupees: Optional[float] = None
    profit_potential_percentage: Optional[float] = None


class MarketplaceMatchingService:
    """Matches buyers, sellers, and investors in the solar marketplace"""
    
    def __init__(self):
        self.users: Dict[str, UserProfile] = {}
        logger.info("MarketplaceMatchingService initialized")
    
    def register_user(self, profile: UserProfile) -> Dict[str, Any]:
        """Register a user for marketplace matching"""
        self.users[profile.user_id] = profile
        logger.info(f"Registered {profile.user_type}: {profile.user_id}")
        return {
            "status": "registered",
            "user_id": profile.user_id,
            "user_type": profile.user_type
        }
    
    def calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate distance between two coordinates using Haversine formula (in km)
        """
        R = 6371  # Earth radius in km
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lon = math.radians(lon2 - lon1)
        
        a = math.sin(delta_lat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        return R * c
    
    def match_seller_to_buyers(self, seller_id: str, limit: int = 10) -> List[MatchingResult]:
        """
        Find best buyers for a seller based on:
        - Proximity (distance)
        - Demand match (buyer need ≈ seller capacity)
        - Financial viability (buyer can pay)
        - Credit rating (buyer reliability)
        """
        if seller_id not in self.users:
            return []
        
        seller = self.users[seller_id]
        if seller.user_type != "seller":
            return []
        
        buyers = [u for u in self.users.values() if u.user_type == "buyer"]
        matches: List[MatchingResult] = []
        
        for buyer in buyers:
            # Distance check
            distance = self.calculate_distance(
                seller.latitude, seller.longitude,
                buyer.latitude, buyer.longitude
            )
            
            if distance > buyer.willing_to_travel_km:
                continue
            
            # Calculate match score
            match_score = self._calculate_buyer_seller_match(seller, buyer, distance)
            
            # Financial feasibility
            feasibility = self._calculate_financial_feasibility(seller, buyer)
            
            # Recommended price
            estimated_system_value = self._estimate_system_value(seller)
            
            # Transaction risk
            risk = self._assess_buyer_risk(buyer)
            
            matches.append(MatchingResult(
                match_id=f"{seller_id}_to_{buyer.user_id}_{int(datetime.now().timestamp())}",
                user_a_id=seller_id,
                user_b_id=buyer.user_id,
                match_score=match_score,
                distance_km=distance,
                compatibility_reason=f"Buyer needs {buyer.demand_kw}kW, seller has {seller.capacity_kw}kW at {distance:.1f}km. Financial score: {buyer.financial_score:.0f}/100",
                financial_feasibility=feasibility,
                transaction_risk=risk,
                recommended_price_rupees=estimated_system_value,
                profit_potential_percentage=feasibility.get("profit_potential", 0)
            ))
        
        # Sort by match score (descending)
        matches.sort(key=lambda m: m.match_score, reverse=True)
        return matches[:limit]
    
    def match_investor_to_sellers(self, investor_id: str, limit: int = 10) -> List[MatchingResult]:
        """
        Find best sellers for an investor based on:
        - Proximity to location of interest
        - System capacity and efficiency (higher ROI potential)
        - System age (newer = lower maintenance cost)
        - Financial match (investor can fund the project)
        - Expected ROI vs investor expectations
        """
        if investor_id not in self.users:
            return []
        
        investor = self.users[investor_id]
        if investor.user_type != "investor":
            return []
        
        sellers = [u for u in self.users.values() if u.user_type == "seller"]
        matches: List[MatchingResult] = []
        
        for seller in sellers:
            # Distance check
            distance = self.calculate_distance(
                investor.latitude, investor.longitude,
                seller.latitude, seller.longitude
            )
            
            if distance > investor.willing_to_travel_km:
                continue
            
            # Calculate match score
            match_score = self._calculate_investor_seller_match(investor, seller, distance)
            
            # Financial feasibility
            feasibility = self._calculate_investor_feasibility(investor, seller)
            
            # Estimated ROI
            roi_potential = self._estimate_roi(seller, investor)
            
            # Transaction risk
            risk = self._assess_seller_risk(seller)
            
            matches.append(MatchingResult(
                match_id=f"{investor_id}_to_{seller.user_id}_{int(datetime.now().timestamp())}",
                user_a_id=investor_id,
                user_b_id=seller.user_id,
                match_score=match_score,
                distance_km=distance,
                compatibility_reason=f"Seller has {seller.capacity_kw}kW system, {seller.system_efficiency:.0f}% efficient, {seller.system_age_years:.1f} years old. ROI potential: {roi_potential:.1f}%",
                financial_feasibility=feasibility,
                transaction_risk=risk,
                recommended_price_rupees=feasibility.get("investment_required", 0),
                profit_potential_percentage=roi_potential
            ))
        
        # Sort by ROI potential (descending)
        matches.sort(key=lambda m: m.profit_potential_percentage or 0, reverse=True)
        return matches[:limit]
    
    def match_buyer_to_sellers(self, buyer_id: str, limit: int = 10) -> List[MatchingResult]:
        """
        Find best sellers for a buyer (reverse of match_seller_to_buyers)
        """
        if buyer_id not in self.users:
            return []
        
        buyer = self.users[buyer_id]
        if buyer.user_type != "buyer":
            return []
        
        sellers = [u for u in self.users.values() if u.user_type == "seller"]
        matches: List[MatchingResult] = []
        
        for seller in sellers:
            distance = self.calculate_distance(
                buyer.latitude, buyer.longitude,
                seller.latitude, seller.longitude
            )
            
            if distance > buyer.willing_to_travel_km:
                continue
            
            # Capacity must match
            if seller.capacity_kw < buyer.demand_kw * 0.8 or seller.capacity_kw > buyer.demand_kw * 1.3:
                continue
            
            match_score = self._calculate_buyer_seller_match(seller, buyer, distance)
            feasibility = self._calculate_financial_feasibility(seller, buyer)
            estimated_system_value = self._estimate_system_value(seller)
            risk = self._assess_seller_risk(seller)
            
            matches.append(MatchingResult(
                match_id=f"{buyer.user_id}_to_{seller.user_id}_{int(datetime.now().timestamp())}",
                user_a_id=buyer_id,
                user_b_id=seller.user_id,
                match_score=match_score,
                distance_km=distance,
                compatibility_reason=f"Seller has {seller.capacity_kw}kW (buyer needs {buyer.demand_kw}kW). Financial match: {buyer.financial_score:.0f}/100. Seller rating: {seller.credit_rating}",
                financial_feasibility=feasibility,
                transaction_risk=risk,
                recommended_price_rupees=estimated_system_value
            ))
        
        matches.sort(key=lambda m: m.match_score, reverse=True)
        return matches[:limit]
    
    def get_nearby_sellers(self, latitude: float, longitude: float, radius_km: float = 25, limit: int = 5) -> List[Dict[str, Any]]:
        """Get nearby sellers within a radius"""
        sellers = []
        for user in self.users.values():
            if user.user_type == "seller":
                distance = self.calculate_distance(latitude, longitude, user.latitude, user.longitude)
                if distance <= radius_km:
                    sellers.append({
                        "seller_id": user.user_id,
                        "capacity_kw": user.capacity_kw,
                        "efficiency": user.system_efficiency,
                        "distance_km": round(distance, 2),
                        "credit_rating": user.credit_rating,
                        "system_age_years": user.system_age_years,
                        "estimated_price": self._estimate_system_value(user),
                        "roi_potential": self._estimate_roi_for_buyer(user)
                    })
        
        sellers.sort(key=lambda s: s["distance_km"])
        return sellers[:limit]
    
    def get_nearby_buyers(self, latitude: float, longitude: float, radius_km: float = 25, limit: int = 5) -> List[Dict[str, Any]]:
        """Get nearby buyers within a radius"""
        buyers = []
        for user in self.users.values():
            if user.user_type == "buyer":
                distance = self.calculate_distance(latitude, longitude, user.latitude, user.longitude)
                if distance <= radius_km:
                    buyers.append({
                        "buyer_id": user.user_id,
                        "demand_kw": user.demand_kw,
                        "budget_rupees": user.budget_rupees,
                        "distance_km": round(distance, 2),
                        "credit_rating": user.credit_rating,
                        "financial_score": user.financial_score
                    })
        
        buyers.sort(key=lambda b: b["distance_km"])
        return buyers[:limit]
    
    def get_nearby_investors(self, latitude: float, longitude: float, radius_km: float = 50, limit: int = 5) -> List[Dict[str, Any]]:
        """Get nearby investors willing to invest in the area"""
        investors = []
        for user in self.users.values():
            if user.user_type == "investor":
                distance = self.calculate_distance(latitude, longitude, user.latitude, user.longitude)
                if distance <= radius_km:
                    investors.append({
                        "investor_id": user.user_id,
                        "available_capital": user.available_balance,
                        "max_investment": user.max_investment,
                        "roi_expectation": user.roi_expectation,
                        "distance_km": round(distance, 2),
                        "credit_rating": user.credit_rating,
                        "financial_score": user.financial_score
                    })
        
        investors.sort(key=lambda i: i["distance_km"])
        return investors[:limit]
    
    # Private helper methods
    
    def _calculate_buyer_seller_match(self, seller: UserProfile, buyer: UserProfile, distance: float) -> float:
        """Calculate match score (0-100) between seller and buyer"""
        score = 100
        
        # Distance penalty (0-20 points)
        distance_score = max(0, 20 - (distance / 25 * 20))
        score -= (20 - distance_score)
        
        # Capacity match (0-30 points)
        if seller.capacity_kw and buyer.demand_kw:
            capacity_ratio = seller.capacity_kw / buyer.demand_kw
            if 0.8 <= capacity_ratio <= 1.3:
                capacity_score = 30
            elif 0.6 <= capacity_ratio <= 1.5:
                capacity_score = 20
            else:
                capacity_score = 10
        else:
            capacity_score = 15
        score -= (30 - capacity_score)
        
        # Financial viability (0-30 points)
        if buyer.budget_rupees and self._estimate_system_value(seller) <= buyer.budget_rupees:
            financial_score = 30
        elif buyer.available_balance >= self._estimate_system_value(seller) * 0.5:
            financial_score = 20
        else:
            financial_score = 5
        score -= (30 - financial_score)
        
        # Credit rating match (0-20 points)
        rating_weight = {"excellent": 20, "good": 15, "fair": 10, "poor": 0}
        rating_score = rating_weight.get(buyer.credit_rating, 5)
        score -= (20 - rating_score)
        
        return max(0, min(100, score))
    
    def _calculate_investor_seller_match(self, investor: UserProfile, seller: UserProfile, distance: float) -> float:
        """Calculate match score between investor and seller"""
        score = 100
        
        # Distance penalty
        distance_score = max(0, 20 - (distance / 50 * 20))
        score -= (20 - distance_score)
        
        # System efficiency & age (0-30 points)
        if seller.system_efficiency and seller.system_age_years:
            efficiency_bonus = seller.system_efficiency * 0.3
            age_penalty = seller.system_age_years * 2
            efficiency_score = min(30, max(5, 30 - age_penalty + (efficiency_bonus - 50)))
        else:
            efficiency_score = 15
        score -= (30 - efficiency_score)
        
        # Financial capacity (0-30 points)
        investment_required = self._estimate_system_value(seller)
        if investor.available_balance >= investment_required and investor.max_investment >= investment_required:
            financial_score = 30
        elif investor.available_balance >= investment_required * 0.7:
            financial_score = 20
        else:
            financial_score = 5
        score -= (30 - financial_score)
        
        # Credit rating (0-20 points)
        rating_weight = {"excellent": 20, "good": 15, "fair": 10, "poor": 0}
        rating_score = rating_weight.get(seller.credit_rating, 5)
        score -= (20 - rating_score)
        
        return max(0, min(100, score))
    
    def _calculate_financial_feasibility(self, seller: UserProfile, buyer: UserProfile) -> Dict[str, Any]:
        """Calculate financial terms between seller and buyer"""
        system_value = self._estimate_system_value(seller)
        
        return {
            "seller_asking_price": system_value,
            "buyer_budget_available": buyer.available_balance,
            "can_afford_full": buyer.available_balance >= system_value,
            "financing_needed": max(0, system_value - buyer.available_balance),
            "suggested_emi_months": 12,
            "monthly_emi": system_value / 12 if system_value > 0 else 0,
            "profit_potential": ((system_value * 0.15) / 12) * 12  # 15% markup potential
        }
    
    def _calculate_investor_feasibility(self, investor: UserProfile, seller: UserProfile) -> Dict[str, Any]:
        """Calculate investment feasibility"""
        investment_required = self._estimate_system_value(seller)
        estimated_annual_revenue = (seller.capacity_kw * 5 * 365 * 5) if seller.capacity_kw else 0  # ₹5/kWh avg
        
        return {
            "investment_required": investment_required,
            "investor_available_capital": investor.available_balance,
            "can_invest": investor.available_balance >= investment_required,
            "estimated_annual_revenue": estimated_annual_revenue,
            "estimated_payback_years": investment_required / (estimated_annual_revenue + 1) if estimated_annual_revenue > 0 else 0,
            "roi_annual_percentage": (estimated_annual_revenue / investment_required * 100) if investment_required > 0 else 0
        }
    
    def _estimate_system_value(self, seller: UserProfile) -> float:
        """Estimate solar system market value based on capacity and condition"""
        if not seller.capacity_kw:
            return 0
        
        base_price_per_kw = 100000  # ₹100,000 per kW (market rate)
        system_value = seller.capacity_kw * base_price_per_kw
        
        # Adjust for system age
        if seller.system_age_years:
            age_depreciation = seller.system_age_years * 0.05  # 5% depreciation per year
            system_value *= (1 - min(age_depreciation, 0.5))  # Max 50% depreciation
        
        # Adjust for efficiency
        if seller.system_efficiency:
            efficiency_premium = (seller.system_efficiency - 75) * 0.001  # Bonus if > 75%
            system_value *= (1 + efficiency_premium)
        
        return system_value
    
    def _estimate_roi(self, seller: UserProfile, investor: UserProfile) -> float:
        """Estimate annual ROI for investor on seller's system"""
        if not seller.capacity_kw:
            return 0
        
        # Assume 5kWh avg production per day per kW, ₹5 per kWh
        annual_kwh = seller.capacity_kw * 5 * 365
        annual_revenue = annual_kwh * 5
        
        investment_cost = self._estimate_system_value(seller)
        roi_percentage = (annual_revenue / investment_cost * 100) if investment_cost > 0 else 0
        
        return min(roi_percentage, 25)  # Cap at 25% realistic max
    
    def _estimate_roi_for_buyer(self, seller: UserProfile) -> float:
        """Estimate ROI if buyer operates the system (savings-based)"""
        if not seller.capacity_kw:
            return 0
        
        # Assume ₹10 per kWh savings vs grid
        daily_savings = seller.capacity_kw * 5 * 10
        annual_savings = daily_savings * 365
        
        system_cost = self._estimate_system_value(seller)
        roi = (annual_savings / system_cost * 100) if system_cost > 0 else 0
        
        return min(roi, 20)
    
    def _assess_buyer_risk(self, buyer: UserProfile) -> str:
        """Assess transaction risk for buyer"""
        score = buyer.financial_score
        
        if buyer.credit_rating == "excellent" and score >= 80:
            return "low"
        elif buyer.credit_rating in ["excellent", "good"] and score >= 60:
            return "medium"
        else:
            return "high"
    
    def _assess_seller_risk(self, seller: UserProfile) -> str:
        """Assess transaction risk for seller"""
        score = seller.financial_score
        
        if seller.credit_rating == "excellent" and score >= 80:
            return "low"
        elif seller.credit_rating in ["excellent", "good"] and score >= 60:
            return "medium"
        else:
            return "high"
