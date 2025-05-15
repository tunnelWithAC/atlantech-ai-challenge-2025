
import { AccessibilityData } from "@/components/AccessibilityResult";

// Mock data for the app
export const mockBuildingData: Record<string, AccessibilityData> = {
  "bldg-1": {
    buildingId: "bldg-1",
    buildingName: "Downtown Office Tower",
    score: 92,
    transportOptions: [
      { type: "subway", line: "Blue Line", frequency: "Every 5 minutes", distance: "1 min walk" },
      { type: "bus", line: "Route 42", frequency: "Every 10 minutes", distance: "2 min walk" },
      { type: "bus", line: "Route 38", frequency: "Every 15 minutes", distance: "3 min walk" },
      { type: "train", line: "Central Station", frequency: "Multiple lines", distance: "8 min walk" }
    ],
    explanation: "This building has excellent accessibility with multiple high-frequency transit options within short walking distance. The subway entrance is directly adjacent to the building, and multiple bus routes stop nearby. The central train station is also within walking distance, providing regional connectivity."
  },
  "bldg-2": {
    buildingId: "bldg-2",
    buildingName: "Riverside Apartments",
    score: 78,
    transportOptions: [
      { type: "bus", line: "Route 17", frequency: "Every 12 minutes", distance: "5 min walk" },
      { type: "tram", line: "Riverside Line", frequency: "Every 15 minutes", distance: "7 min walk" },
      { type: "train", line: "Riverside Station", frequency: "Hourly service", distance: "15 min walk" }
    ],
    explanation: "The Riverside Apartments offer very good transit access with regular bus service and a tram line within walking distance. The tram provides direct access to downtown without transfers. While the regional train station is a bit further away, it provides additional connectivity options for longer trips."
  },
  "bldg-3": {
    buildingId: "bldg-3",
    buildingName: "Central Library",
    score: 85,
    transportOptions: [
      { type: "bus", line: "Route 10", frequency: "Every 8 minutes", distance: "1 min walk" },
      { type: "bus", line: "Route 22", frequency: "Every 10 minutes", distance: "1 min walk" },
      { type: "subway", line: "Red Line", frequency: "Every 7 minutes", distance: "6 min walk" }
    ],
    explanation: "The Central Library is very well served by public transit, with multiple high-frequency bus routes stopping directly in front of the building. The subway station is within a short walking distance, offering fast connections to all parts of the city."
  },
  "bldg-4": {
    buildingId: "bldg-4",
    buildingName: "City Hospital",
    score: 88,
    transportOptions: [
      { type: "bus", line: "Hospital Express", frequency: "Every 10 minutes", distance: "At entrance" },
      { type: "bus", line: "Route 5", frequency: "Every 12 minutes", distance: "2 min walk" },
      { type: "subway", line: "Green Line", frequency: "Every 6 minutes", distance: "5 min walk" }
    ],
    explanation: "City Hospital features excellent transit accessibility with a dedicated bus stop at its main entrance. Multiple regular bus routes and a nearby subway station ensure easy access for patients, visitors, and staff at all hours. The Hospital Express bus provides direct service to major transit hubs."
  },
  "bldg-5": {
    buildingId: "bldg-5",
    buildingName: "University Campus Center",
    score: 82,
    transportOptions: [
      { type: "bus", line: "Campus Shuttle", frequency: "Every 7 minutes", distance: "At entrance" },
      { type: "bus", line: "Route 33", frequency: "Every 15 minutes", distance: "3 min walk" },
      { type: "tram", line: "University Line", frequency: "Every 10 minutes", distance: "8 min walk" }
    ],
    explanation: "The University Campus Center has very good transit access, particularly with the frequent campus shuttle that connects to major transit hubs. The city bus and nearby tram line provide additional options for traveling throughout the city. Peak service increases during academic terms."
  },
  "bldg-6": {
    buildingId: "bldg-6",
    buildingName: "Tech Innovation Hub",
    score: 65,
    transportOptions: [
      { type: "bus", line: "Route 88", frequency: "Every 20 minutes", distance: "7 min walk" },
      { type: "train", line: "North Station", frequency: "Every 30 minutes", distance: "15 min walk" }
    ],
    explanation: "The Tech Innovation Hub has good transit accessibility, though options are more limited than downtown locations. The bus service is regular but less frequent, and the nearest train station requires a longer walk. Many employees use the company's private shuttle service to supplement public transit options."
  },
  "bldg-7": {
    buildingId: "bldg-7",
    buildingName: "Westside Shopping Mall",
    score: 12,
    transportOptions: [
      { type: "bus", line: "Mall Express", frequency: "Every 15 minutes", distance: "At entrance" },
      { type: "bus", line: "Route 45", frequency: "Every 20 minutes", distance: "5 min walk" },
      { type: "bus", line: "Route 46", frequency: "Every 25 minutes", distance: "5 min walk" }
    ],
    explanation: "Westside Shopping Mall offers good transit accessibility with several bus routes serving the property directly. The dedicated Mall Express connects to the downtown transit hub and runs extended hours during shopping peak times. Weekend service is slightly reduced but maintains good coverage."
  }
};

// Simulate API call delay
// export const fetchBuildingAccessibility = (buildingId: string): Promise<AccessibilityData> => {
//   return new Promise((resolve) => {
//     setTimeout(() => {
//       resolve(mockBuildingData[buildingId]);
//     }, 1500);
//   });
// };
