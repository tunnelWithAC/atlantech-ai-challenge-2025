import { AccessibilityData } from "@/components/AccessibilityResult";

const BUILDING_NAMES: Record<string, string> = {
  "bldg-1": "Downtown Office Tower",
  "bldg-2": "Riverside Apartments",
  "bldg-3": "Central Library",
  "bldg-4": "City Hospital",
  "bldg-5": "University Campus Center",
  "bldg-6": "Tech Innovation Hub",
  "bldg-7": "Westside Shopping Mall"
};

interface OfficeResponse {
  content: string;
  office_name: string;
  scores: {
    barna: number;
    knocknacarra: number;
    oranmore: number;
  };
}

// API client function to fetch building data
export const fetchBuildingAccessibility = async (buildingId: string): Promise<AccessibilityData> => {
  try {
    // Add timestamp to URL to prevent caching
    const timestamp = new Date().getTime();
    const response = await fetch(`http://localhost:5001/prompt?office_name=${buildingId}&_=${timestamp}`, {
      headers: {
        'Accept': 'application/json',
        'Cache-Control': 'no-cache, no-store, must-revalidate',
        'Pragma': 'no-cache',
        'Expires': '0'
      }
    });
    
    if (!response.ok) {
      throw new Error(`Failed to fetch building data: ${response.status} ${response.statusText}`);
    }
    
    const data: OfficeResponse = await response.json();
    
    // Transform the API response into AccessibilityData format
    return {
      buildingId: buildingId,
      buildingName: BUILDING_NAMES[buildingId] || buildingId,
      score: Math.round((data.scores.barna + data.scores.knocknacarra + data.scores.oranmore) / 3 * 10), // Average score out of 100
      explanation: data.content
    };
  } catch (error) {
    console.error('Error fetching building data:', error);
    throw error;
  }
};
