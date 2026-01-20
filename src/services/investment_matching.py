"""
Investment Matching Algorithm
Finds best (Buyer → Host → Industry) matches using AI
"""

import numpy as np
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class InvestmentMatcher:
    """
    AI-powered investment matching algorithm
    Matches buyers with optimal host locations near industries
    """

    def __init__(self):
        self.weights = {
            'distance_score': 0.25,
            'roi_score': 0.30,
            'risk_score': 0.20,
            'demand_match_score': 0.15,
            'location_score': 0.10,
        }

    def find_best_matches(
        self,
        buyer_budget: float,
        buyer_location: Dict[str, str],
        buyer_risk_tolerance: str,  # 'low', 'medium', 'high'
        host_spaces: List[Dict],
        industries: List[Dict],
        max_results: int = 5,
    ) -> List[Dict]:
        """
        Find top investment opportunities for buyer

        Args:
            buyer_budget: Investment budget in INR
            buyer_location: {'city': str, 'state': str}
            buyer_risk_tolerance: Risk appetite
            host_spaces: List of available host spaces
            industries: List of industries needing energy
            max_results: Number of results to return

        Returns:
            List of investment opportunities with scores
        """
        opportunities = []

        for host in host_spaces:
            # Check if budget sufficient
            required_investment = host['available_capacity_kw'] * 550000  # ₹5.5L per kW
            if required_investment > buyer_budget * 1.2:  # Allow 20% buffer
                continue

            # Find nearby industries
            nearby_industries = self._find_nearby_industries(host, industries)

            if not nearby_industries:
                continue

            # Calculate best industry match
            best_industry = max(
                nearby_industries,
                key=lambda ind: self._calculate_industry_score(host, ind),
            )

            # Calculate comprehensive score
            opportunity = self._create_opportunity(
                buyer_budget, buyer_location, buyer_risk_tolerance, host, best_industry
            )

            opportunities.append(opportunity)

        # Sort by match score
        opportunities.sort(key=lambda x: x['ai_match_score'], reverse=True)

        return opportunities[:max_results]

    def _find_nearby_industries(
        self, host: Dict, industries: List[Dict], max_distance_km: float = 50
    ) -> List[Dict]:
        """Find industries within reasonable distance of host"""
        nearby = []

        for industry in industries:
            distance = self._calculate_distance(
                host['latitude'],
                host['longitude'],
                industry.get('latitude', 0),
                industry.get('longitude', 0),
            )

            if distance <= max_distance_km:
                industry['distance_km'] = distance
                nearby.append(industry)

        return nearby

    def _calculate_distance(
        self, lat1: float, lon1: float, lat2: float, lon2: float
    ) -> float:
        """Calculate distance between two points using Haversine formula"""
        from math import radians, cos, sin, asin, sqrt

        # Convert to radians
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        c = 2 * asin(sqrt(a))
        r = 6371  # Earth radius in km

        return c * r

    def _calculate_industry_score(self, host: Dict, industry: Dict) -> float:
        """Score industry match quality"""
        score = 0.0

        # Distance score (closer is better)
        distance_km = industry.get('distance_km', 50)
        distance_score = max(0, 1 - (distance_km / 50))
        score += distance_score * 0.4

        # Demand match score
        host_production = host['available_capacity_kw'] * 150 * 30  # Monthly kWh
        industry_demand = industry['daily_energy_demand_kwh'] * 30  # Monthly kWh

        demand_ratio = min(host_production / industry_demand, 1.0)
        score += demand_ratio * 0.3

        # Price match score
        host_expected_price = 8.0  # ₹8/kWh
        industry_max_price = industry.get('max_price_per_kwh', 10.0)
        if industry_max_price >= host_expected_price:
            score += 0.3

        return score

    def _create_opportunity(
        self,
        buyer_budget: float,
        buyer_location: Dict,
        buyer_risk_tolerance: str,
        host: Dict,
        industry: Dict,
    ) -> Dict:
        """Create detailed opportunity with all scores"""

        # Calculate financial metrics
        capacity_kw = min(host['available_capacity_kw'], buyer_budget / 550000)
        panel_price = capacity_kw * 500000
        installation_cost = capacity_kw * 50000
        total_investment = panel_price + installation_cost

        monthly_production_kwh = capacity_kw * 150
        price_per_kwh = industry.get('max_price_per_kwh', 8.0)
        monthly_revenue = monthly_production_kwh * price_per_kwh

        buyer_share = 0.85  # 85%
        host_rent = host['monthly_rent_per_kw'] * capacity_kw
        platform_fee = monthly_revenue * 0.10

        monthly_profit = (monthly_revenue * buyer_share) - host_rent - platform_fee
        annual_return = monthly_profit * 12
        roi_percentage = (annual_return / total_investment) * 100
        payback_months = int(total_investment / monthly_profit) if monthly_profit > 0 else 999

        # Calculate individual scores
        distance_score = self._score_distance(host, buyer_location)
        roi_score = self._score_roi(roi_percentage)
        risk_score = self._score_risk(host, buyer_risk_tolerance)
        demand_match_score = self._score_demand_match(host, industry)
        location_score = self._score_location(host, industry)

        # Weighted total score
        ai_match_score = (
            distance_score * self.weights['distance_score']
            + roi_score * self.weights['roi_score']
            + risk_score * self.weights['risk_score']
            + demand_match_score * self.weights['demand_match_score']
            + location_score * self.weights['location_score']
        )

        ai_match_score = int(ai_match_score * 100)  # Convert to 0-100 scale

        # Determine if AI recommended (score >= 80)
        is_ai_recommended = ai_match_score >= 80

        return {
            'id': host['id'],
            'host_id': host['host_id'],
            'host_name': host['host_name'],
            'host_rating': host.get('host_rating', 4.5),
            'location': f"{host['city']}, {host['state']}",
            'latitude': host.get('latitude', 0),
            'longitude': host.get('longitude', 0),
            'available_capacity_kw': capacity_kw,
            'panel_price': panel_price,
            'installation_cost': installation_cost,
            'total_investment': total_investment,
            'estimated_monthly_production_kwh': monthly_production_kwh,
            'estimated_monthly_revenue': monthly_revenue,
            'estimated_monthly_profit': monthly_profit,
            'estimated_roi_percentage': round(roi_percentage, 2),
            'payback_period_months': payback_months,
            'nearby_industries': 1,
            'distance_km': industry.get('distance_km', 5),
            'risk_score': int((1 - risk_score) * 100),  # Invert (lower is better)
            'ai_match_score': ai_match_score,
            'property_images': host.get('property_images', []),
            'is_ai_recommended': is_ai_recommended,
            'industry_id': industry['id'],
            'industry_name': industry['company_name'],
            'industry_price_per_kwh': price_per_kwh,
        }

    def _score_distance(self, host: Dict, buyer_location: Dict) -> float:
        """Score based on distance from buyer"""
        if host['city'] == buyer_location.get('city'):
            return 1.0  # Same city
        elif host['state'] == buyer_location.get('state'):
            return 0.7  # Same state
        else:
            return 0.4  # Different state

    def _score_roi(self, roi_percentage: float) -> float:
        """Score based on ROI (higher is better)"""
        # Normalize ROI (typical range 12-25%)
        if roi_percentage >= 20:
            return 1.0
        elif roi_percentage >= 15:
            return 0.8
        elif roi_percentage >= 12:
            return 0.6
        else:
            return 0.3

    def _score_risk(self, host: Dict, buyer_risk_tolerance: str) -> float:
        """Score based on risk factors"""
        risk_factors = []

        # Has structural certificate (reduces risk)
        if host.get('has_structural_certificate'):
            risk_factors.append(0.9)
        else:
            risk_factors.append(0.6)

        # Host rating
        rating = host.get('host_rating', 4.0)
        risk_factors.append(rating / 5.0)

        # Near industry (reduces risk)
        if host.get('is_near_industry'):
            risk_factors.append(0.9)
        else:
            risk_factors.append(0.7)

        base_risk_score = np.mean(risk_factors)

        # Adjust for buyer's risk tolerance
        if buyer_risk_tolerance == 'low':
            # Conservative buyers need higher scores
            return base_risk_score * 1.0
        elif buyer_risk_tolerance == 'medium':
            return base_risk_score * 0.95
        else:  # high
            return base_risk_score * 0.9

    def _score_demand_match(self, host: Dict, industry: Dict) -> float:
        """Score based on supply-demand match"""
        host_supply = host['available_capacity_kw'] * 150 * 30  # Monthly kWh
        industry_demand = industry['daily_energy_demand_kwh'] * 30

        ratio = host_supply / industry_demand if industry_demand > 0 else 0

        if 0.8 <= ratio <= 1.2:
            return 1.0  # Perfect match
        elif 0.5 <= ratio <= 1.5:
            return 0.8  # Good match
        elif 0.3 <= ratio <= 2.0:
            return 0.6  # Acceptable match
        else:
            return 0.4  # Poor match

    def _score_location(self, host: Dict, industry: Dict) -> float:
        """Score based on host-industry distance"""
        distance_km = industry.get('distance_km', 50)

        if distance_km <= 5:
            return 1.0
        elif distance_km <= 15:
            return 0.8
        elif distance_km <= 30:
            return 0.6
        else:
            return 0.4


# Singleton instance
investment_matcher = InvestmentMatcher()
