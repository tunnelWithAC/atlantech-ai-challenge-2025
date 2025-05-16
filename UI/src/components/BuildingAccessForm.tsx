import { useState } from "react";
import { 
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle 
} from "@/components/ui/card";
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Button } from "@/components/ui/button";

interface BuildingAccessFormProps {
  onSubmit: (buildingId: string) => void;
  isLoading: boolean;
}

const BUILDINGS = [
  { id: "bldg-1", name: "Downtown Office Tower" },
  { id: "bldg-2", name: "Riverside Apartments" },
  { id: "bldg-3", name: "Central Library" },
  { id: "bldg-4", name: "City Hospital" },
  { id: "bldg-5", name: "University Campus Center" },
  { id: "bldg-6", name: "Tech Innovation Hub" },
  { id: "bldg-7", name: "Westside Shopping Mall" }
];

export const BuildingAccessForm = ({ onSubmit, isLoading }: BuildingAccessFormProps) => {
  const [selectedBuilding, setSelectedBuilding] = useState<string>("");
  
  const handleSubmit = () => {
    if (selectedBuilding) {
      onSubmit(selectedBuilding);
    }
  };
  
  return (
    <Card className="w-full mx-auto">
      <CardHeader>
        <CardTitle className="text-xl">Check Building Accessibility</CardTitle>
        <CardDescription>
          Select a building to check public transport accessibility options
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          <div className="space-y-2">
            <Select value={selectedBuilding} onValueChange={setSelectedBuilding}>
              <SelectTrigger className="w-full">
                <SelectValue placeholder="Select a building" />
              </SelectTrigger>
              <SelectContent>
                <SelectGroup>
                  {BUILDINGS.map((building) => (
                    <SelectItem key={building.id} value={building.id}>
                      {building.name}
                    </SelectItem>
                  ))}
                </SelectGroup>
              </SelectContent>
            </Select>
          </div>
        </div>
      </CardContent>
      <CardFooter>
        <Button 
          className="w-full" 
          onClick={handleSubmit} 
          disabled={!selectedBuilding || isLoading}
        >
          {isLoading ? "Calculating..." : "Check Accessibility"}
        </Button>
      </CardFooter>
    </Card>
  );
};
