"""
Investment Matching Routes - API endpoints for ML-powered opportunity matching
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional
import sys
import os

# Add services directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from services.investment_matching import InvestmentMatcher

router = APIRouter(prefix="/api/v1/ml/matching", tags=["Investment Matching"])

# Request/Response models
class Location(BaseModel):
    city: str
    state: str
    lat: float
    lng: float

class HostSpace(BaseModel):
    id: int
    host_id: int
    host_name: str
    property_type: str
    available_capacity_kw: float
    total_capacity_kw: float
    city: str
    state: str
    lat: float
    lng: float
    monthly_rent_per_kw: float
    has_structural_certificate: bool
    property_rating: float
    property_images: List[str]

class Industry(BaseModel):
    id: int
    company_name: str
    industry_type: str
    daily_energy_demand_kwh: float
    max_price_per_kwh: float
    city: str
    state: str
    lat: float
    lng: float

class MatchingRequest(BaseModel):
    buyer_budget: float
    buyer_location: Location
    buyer_risk_tolerance: str = "medium"  # low, medium, high
    host_spaces: List[HostSpace]
    industries: List[Industry]
    max_results: int = 10

class OpportunityResponse(BaseModel):
    opportunity_id: str
    host_id: int
    host_name: str
    location: str
    capacity_kw: float
    panel_price: float
    installation_cost: float
    total_investment: float
    monthly_production_kwh: float
    monthly_revenue: float
    buyer_share: float
    host_rent: float
    platform_fee: float
    net_monthly_profit: float
    annual_return: float
    roi_percentage: float
    payback_months: float
    distance_km: float
    risk_score: float
    ai_match_score: float
    is_ai_recommended: bool
    industry: Optional[dict] = None
    property_images: List[str]

# Initialize matcher
matcher = InvestmentMatcher()

@router.post("/find-opportunities", response_model=List[OpportunityResponse])
async def find_best_opportunities(request: MatchingRequest):
    """
    Find best investment opportunities using ML-powered matching algorithm
    
    Parameters:
    - buyer_budget: Available budget for investment
    - buyer_location: Buyer's city, state, lat, lng
    - buyer_risk_tolerance: low, medium, or high
    - host_spaces: List of available host properties
    - industries: List of potential industrial buyers
    - max_results: Maximum number of opportunities to return
    
    Returns list of opportunities sorted by AI match score
    """
    try:
        # Convert Pydantic models to dicts
        buyer_loc = {
            'city': request.buyer_location.city,
            'state': request.buyer_location.state,
            'lat': request.buyer_location.lat,
            'lng': request.buyer_location.lng,
        }
        
        host_spaces = [host.dict() for host in request.host_spaces]
        industries = [industry.dict() for industry in request.industries]
        
        # Find best matches
        opportunities = matcher.find_best_matches(
            buyer_budget=request.buyer_budget,
            buyer_location=buyer_loc,
            buyer_risk_tolerance=request.buyer_risk_tolerance,
            host_spaces=host_spaces,
            industries=industries,
            max_results=request.max_results
        )
        
        # Convert to response format
        response = []
        for opp in opportunities:
            response.append(OpportunityResponse(**opp))
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Matching error: {str(e)}")

@router.post("/score-opportunity")
async def score_single_opportunity(
    buyer_budget: float,
    buyer_location: Location,
    buyer_risk_tolerance: str,
    host_space: HostSpace,
    industries: List[Industry]
):
    """
    Score a single investment opportunity
    
    Useful for re-scoring when data changes or for detailed analysis
    """
    try:
        buyer_loc = {
            'city': buyer_location.city,
            'state': buyer_location.state,
            'lat': buyer_location.lat,
            'lng': buyer_location.lng,
        }
        
        host = host_space.dict()
        industries_list = [industry.dict() for industry in industries]
        
        # Find best match for this single host
        opportunities = matcher.find_best_matches(
            buyer_budget=buyer_budget,
            buyer_location=buyer_loc,
            buyer_risk_tolerance=buyer_risk_tolerance,
            host_spaces=[host],
            industries=industries_list,
            max_results=1
        )
        
        if not opportunities:
            raise HTTPException(status_code=404, detail="No suitable match found")
        
        return OpportunityResponse(**opportunities[0])
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scoring error: {str(e)}")

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Investment Matching ML Service",
        "version": "1.0.0"
    }
