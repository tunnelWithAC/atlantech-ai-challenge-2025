import { BUILDING_ACCESSIBILITY_API } from "@/constants";
import { useState } from "react"
import { useToast } from "./use-toast";

interface ScoreData {
  scores: {
    [key: string]: number;
  };
}

export const useBuildingAccessibility = () => {
  const { toast } = useToast();

    const [ data, setData] = useState<ScoreData | null>(null);
    const [ isLoading, setIsLoading] = useState<boolean>(false);
    const [ error, setError] = useState<Error | null>(null);

    const getBuildingAccessibility = async (buildingName: string) => {
        setIsLoading(true);
        setError(null);
        
        try {
            const response = await fetch(`${BUILDING_ACCESSIBILITY_API}?office_name=${buildingName}`);
            
            if (!response.ok) {
                throw new Error(`API error: ${response.status}`);
            }
            
            const jsonData = await response.json();
            setData(jsonData);
        } catch(error) {
            setError(error as Error);
            console.error("Failed to fetch building data:", error);
            toast({
                title: "Error",
                description: "Failed to load building accessibility data. Please try again.",
                variant: "destructive"
            });
        } finally {
            setIsLoading(false);
        }
    }

    return {
        data, 
        isLoading, 
        error, 
        getBuildingAccessibility,
    }
}