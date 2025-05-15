
import { useEffect, useState } from "react";
import { 
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle 
} from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";

export interface TransportOption {
  type: "bus" | "train" | "subway" | "tram";
  line: string;
  frequency: string;
  distance: string;
}

export interface AccessibilityData {
  buildingId: string;
  buildingName: string;
  score: number;
  transportOptions: TransportOption[];
  explanation: string;
}

interface AccessibilityResultProps {
  data: AccessibilityData | null;
}

const getBadgeClass = (type: TransportOption["type"]) => {
  switch (type) {
    case "bus": return "transit-badge transit-badge-bus";
    case "train": return "transit-badge transit-badge-train";
    case "subway": return "transit-badge transit-badge-subway";
    case "tram": return "transit-badge transit-badge-tram";
    default: return "transit-badge";
  }
};

const getScoreText = (score: number) => {
  if (score >= 90) return "Excellent";
  if (score >= 75) return "Very Good";
  if (score >= 60) return "Good";
  if (score >= 40) return "Fair";
  return "Limited";
};

const getScoreColorClass = (score: number) => {
  if (score >= 90) return "text-green-600";
  if (score >= 75) return "text-emerald-600";
  if (score >= 60) return "text-blue-600";
  if (score >= 40) return "text-amber-600";
  return "text-red-600";
};

export const AccessibilityResult = ({ data }: AccessibilityResultProps) => {
  const [progressValue, setProgressValue] = useState(0);
  
  useEffect(() => {
    if (data) {
      // Animate progress bar
      setProgressValue(0);
      const timer = setTimeout(() => {
        setProgressValue(data.score);
      }, 100);
      return () => clearTimeout(timer);
    }
  }, [data]);
  
  if (!data) return null;
  
  return (
    <Card className="w-full max-w-2xl mx-auto overflow-hidden">
      <CardHeader className="bg-muted pb-2">
        <CardTitle>
          <span>{data.buildingName}</span>
          
        </CardTitle>
        <CardDescription className="flex justify-between items-center">Transport Connectivity Score:
          <span className={`text-2xl font-bold ${getScoreColorClass(data.score)}`}>
            {getScoreText(data.score)}
          </span></CardDescription>
      </CardHeader>
      <CardContent className="pt-6 space-y-4">
        <div className="space-y-2">
          <div className="flex justify-between text-sm">
            <span>Limited</span>
            <span>Excellent</span>
          </div>
          <Progress value={progressValue} className="h-3" />
          <div className="text-right text-sm font-medium">
            {data.score}/100
          </div>
        </div>
        
        {/* <div className="space-y-3">
          <h3 className="font-medium text-lg">Transport Options</h3>
          <ul className="space-y-3">
            {data.transportOptions.map((option, index) => (
              <li key={index} className="flex items-center p-2 bg-muted/50 rounded-md">
                <div className={getBadgeClass(option.type)}>
                  {option.type.charAt(0).toUpperCase() + option.type.slice(1)}
                </div>
                <div className="ml-2">
                  <div className="font-medium">{option.line}</div>
                  <div className="text-sm text-muted-foreground">
                    {option.frequency} • {option.distance}
                  </div>
                </div>
              </li>
            ))}
          </ul>
        </div> */}
        
        <div className="pt-2">
          <h3 className="font-medium text-lg mb-2">Analysis</h3>
          <p className="text-muted-foreground">{data.explanation}</p>
        </div>
      </CardContent>
    </Card>
  );
};
